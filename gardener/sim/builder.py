"""Build the iCub articulation link by link, so joint groups can have different drive settings.

Why this exists: threepp's load_articulation applies ONE stiffness, damping and torque limit to every
joint, and ignores per-joint <limit effort> and <dynamics damping> in the URDF (checked by test).
Articulation.add_link does take per-joint settings, so we rebuild the same robot with it.

Two steps, because PhysX allows only one world per process:
  1. A short subprocess loads the URDF normally and saves every link's collider, mass and joint to
     <urdf>.shapes.json (kept next to the URDF, rebuilt when the URDF changes).
  2. build_articulation() reads that file and calls add_link for each body.

Limitation: only colliders are built, not the visual meshes. View this robot with --colliders.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import threepp as tp


def _quat_to_matrix(w, x, y, z):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _urdf_frames(root):
    """World pose (4x4) of every URDF link at zero joint angles, base at the origin."""
    joints = {j.find("child").get("link"): j for j in root.findall("joint")}
    frames = {}

    def rpy(r, p, y):
        cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
        return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                         [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                         [-sp, cp * sr, cp * cr]])

    def frame(name):
        if name in frames:
            return frames[name]
        T = np.eye(4)
        if name in joints:
            j = joints[name]
            o = j.find("origin")
            local = np.eye(4)
            if o is not None:
                local[:3, :3] = rpy(*[float(v) for v in o.get("rpy", "0 0 0").split()])
                local[:3, 3] = [float(v) for v in o.get("xyz", "0 0 0").split()]
            T = frame(j.find("parent").get("link")) @ local
        frames[name] = T
        return T

    for link in root.findall("link"):
        frame(link.get("name"))
    return frames, joints


def _export_shapes(urdf_path: str, out_path: str) -> None:
    """Runs in the subprocess: load the URDF normally and save what add_link needs."""
    root = ET.parse(urdf_path).getroot()
    frames, joints = _urdf_frames(root)
    world = tp.PhysxWorld(gravity=tp.Vector3(0, 0, -9.81), fixed_timestep=0.002, max_substeps=20)
    art, meshes, joint_names = world.load_articulation(urdf_path, fixed_base=True, default_density=1200.0,
                                                       stiffness=400.0, damping=40.0, max_force=30.0,
                                                       drive_limits_are_forces=True)
    world.step(1e-4)
    links = list(art.links)
    if len(links) != len(joint_names) + 1 or len(meshes) != len(links):
        raise RuntimeError(f"unexpected sizes: {len(links)} links, {len(meshes)} meshes, {len(joint_names)} joints")
    by_joint = {j.get("name"): (child, j) for child, j in joints.items()}

    def ancestor(link_name):
        """Nearest parent that is its own PhysX body (fixed joints are welded away)."""
        j = joints[link_name]
        parent = j.find("parent").get("link")
        while parent in joints and joints[parent].get("type") == "fixed":
            parent = joints[parent].find("parent").get("link")
        return parent

    out = []
    for i, (link, mesh) in enumerate(zip(links, meshes)):
        if i == 0:
            name, joint = next(l.get("name") for l in root.findall("link") if l.get("name") not in joints), None
        else:
            name, joint = by_joint[joint_names[i - 1]]
        # Check the assumed order: the link found by name must be this very body.
        ref, a, b = art.link(name), link.position, None
        b = ref.position
        if abs(a.x - b.x) + abs(a.y - b.y) + abs(a.z - b.z) > 1e-9:
            raise RuntimeError(f"link order assumption failed at {name}")
        mesh.update_matrix_world()
        p, q = mesh.position, mesh.quaternion
        pos, R = np.array([p.x, p.y, p.z]), _quat_to_matrix(q.w, q.x, q.y, q.z)
        v = np.asarray(mesh.geometry.get_attribute("position"), dtype=np.float64).reshape(-1, 3)
        lo, hi = v.min(0), v.max(0)
        extent, center = hi - lo, (hi + lo) / 2
        kind = type(mesh.geometry).__name__
        entry = {"name": name, "mass": link.mass, "quat": [q.w, q.x, q.y, q.z]}
        if kind == "SphereGeometry":
            entry.update(kind="sphere", radius=float(extent.max() / 2), pos=pos.tolist())
        elif kind == "CapsuleGeometry":
            radius = float(min(extent) / 2)
            entry.update(kind="capsule", radius=radius, length=float(max(extent) - 2 * radius), pos=pos.tolist())
        else:  # threepp collides meshes as their bounding box
            entry.update(kind="box", size=extent.tolist(), pos=(pos + R @ center).tolist())
        if joint is not None:
            T = frames[name]
            ax = joint.find("axis")
            axis = np.array([float(x) for x in ax.get("xyz").split()]) if ax is not None else np.array([1.0, 0, 0])
            lim = joint.find("limit")
            entry.update(joint=joint.get("name"), parent=ancestor(name), anchor=T[:3, 3].tolist(),
                         axis=(T[:3, :3] @ axis).tolist(), lower=float(lim.get("lower")), upper=float(lim.get("upper")))
        out.append(entry)
    with open(out_path, "w") as f:
        json.dump(out, f)


def load_shapes(urdf_path: str) -> list[dict]:
    """The saved description of the robot, regenerating it if the URDF is newer."""
    out_path = urdf_path + ".shapes.json"
    if not os.path.exists(out_path) or os.path.getmtime(out_path) < os.path.getmtime(urdf_path):
        subprocess.run([sys.executable, os.path.abspath(__file__), urdf_path, out_path], check=True)
    with open(out_path) as f:
        return json.load(f)


def build_articulation(world, urdf_path: str, base_position, drive_for, solver_iterations: int = 12) -> tuple:
    """Build the robot in `world`. drive_for(joint name) returns (stiffness, damping, max_force).

    Returns (articulation, meshes, joint_names, links_by_name, meshes_by_name).
    """
    base = np.array(base_position, dtype=float)
    art = world.create_articulation(fixed_base=True, solver_position_iterations=int(solver_iterations),
                                    disable_self_collision=True, drive_limits_are_forces=True)
    links, meshes, names = {}, {}, []
    for e in load_shapes(urdf_path):
        if e["kind"] == "sphere":
            geometry, volume = tp.SphereGeometry(e["radius"]), 4 / 3 * math.pi * e["radius"] ** 3
        elif e["kind"] == "capsule":
            r, ln = e["radius"], e["length"]
            geometry, volume = tp.CapsuleGeometry(r, ln), math.pi * r * r * ln + 4 / 3 * math.pi * r ** 3
        else:
            s = e["size"]
            geometry, volume = tp.BoxGeometry(*s), s[0] * s[1] * s[2]
        mesh = tp.Mesh(geometry, tp.MeshStandardMaterial())
        mesh.position.set(*map(float, np.array(e["pos"]) + base))
        w, x, y, z = e["quat"]
        mesh.quaternion.set(x, y, z, w)
        density = e["mass"] / volume  # add_link takes density; this reproduces the saved mass
        if "joint" not in e:
            link = art.add_link(mesh, density=density)
        else:
            k, d, limit = drive_for(e["joint"])
            link = art.add_link(mesh, parent=links[e["parent"]], density=density, axis=e["axis"],
                                anchor=(np.array(e["anchor"]) + base).tolist(), lower=e["lower"], upper=e["upper"],
                                stiffness=k, damping=d, max_force=limit)
            names.append(e["joint"])
        links[e["name"]] = link
        meshes[e["name"]] = mesh
    art.finalize()
    return art, list(meshes.values()), names, links, meshes


if __name__ == "__main__":
    _export_shapes(sys.argv[1], sys.argv[2])
