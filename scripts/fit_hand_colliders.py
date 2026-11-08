#!/usr/bin/env python3
"""Give iCub's right hand grasp-friendly collision shapes.

threepp turns each link's collision mesh into a box (its bounding box), so
the fingers are blocks and the palm is one big block. This script replaces,
in a copy of the URDF:
  - every right-hand finger segment's collision with a CAPSULE fitted to it
    (URDF <cylinder>, which threepp builds as a capsule along the link's z),
  - the palm's collision with a tight, oriented BOX.

How the fit works: load the robot in PhysX, take each part's collision mesh
exactly as threepp loaded it, express its vertices in the link's own frame,
and find the segment's main direction with PCA (principal component
analysis: the direction along which the points are most spread out).

Usage:
  python scripts/fit_hand_colliders.py \
      --src ~/icub-assets/iCub/robots/iCubGazeboV2_10/gardener.urdf \
      --out ~/icub-assets/iCub/robots/iCubGazeboV2_10/gardener_hand.urdf
"""
import argparse
import math
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import threepp as tp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gardener.sim.robot import FINGERS, ICub, make_world  # noqa: E402

HAND_LINKS = ["r_hand"] + [f"r_hand_{f}_{k}" for f in FINGERS for k in range(4)]
RADIUS_PERCENTILE = 80   # capsule radius = this percentile of the points' distance to the axis
# Arm links that also get a capsule, with their own radius percentile. The forearm's bounding box
# (146 x 109 x 74 mm) stops the arm reaching over a table; its mesh is close to a 43 mm radius capsule.
ARM_CAPSULES = {"r_forearm": 95, "r_upper_arm": 95}
BOX_PERCENTILE = 97      # palm box half-sizes = this percentile of |coordinate| along each axis
BASE_POSITION = (0.0, 0.0, 0.05)  # must match ICub's default base_position


def quat_to_matrix(w, x, y, z):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_rpy(R):
    """URDF rpy: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    pitch = math.asin(-max(-1.0, min(1.0, R[2, 0])))
    if abs(math.cos(pitch)) > 1e-6:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:  # gimbal lock
        roll, yaw = math.atan2(-R[1, 2], R[1, 1]), 0.0
    return roll, pitch, yaw


def frame_with_z(axis):
    """A right-handed rotation whose z column is `axis`."""
    z = axis / np.linalg.norm(axis)
    helper = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
    x = np.cross(helper, z)
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def rpy_to_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def urdf_link_frames(root_el, base_position):
    """World pose (4x4) of every URDF link at the zero joint configuration.

    Needed because a PhysX link's pose is its centre-of-mass/inertia frame,
    which is offset and rotated from the URDF link frame that <collision>
    origins refer to.
    """
    joints = {j.find("child").get("link"): j for j in root_el.findall("joint")}
    frames = {}

    def frame(name):
        if name in frames:
            return frames[name]
        if name not in joints:  # the root link
            T = np.eye(4)
            T[:3, 3] = base_position
        else:
            j = joints[name]
            o = j.find("origin")
            xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
            rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
            local = np.eye(4)
            local[:3, :3] = rpy_to_matrix(*rpy)
            local[:3, 3] = xyz
            T = frame(j.find("parent").get("link")) @ local
        frames[name] = T
        return T

    for l in root_el.findall("link"):
        frame(l.get("name"))
    return frames


def points_in_link_frame(mesh, T_link):
    """The collider mesh's vertices, expressed in the URDF link frame T_link (4x4)."""
    mesh.update_matrix_world()
    v = np.asarray(mesh.geometry.get_attribute("position"), dtype=np.float64).reshape(-1, 3)
    world = np.array([[q.x, q.y, q.z] for q in
                      (mesh.local_to_world(tp.Vector3(*map(float, p))) for p in v)])
    R, p = T_link[:3, :3], T_link[:3, 3]
    return (world - p) @ R  # = R^T (world - p) for each row


def fit_capsule(pts, radius_percentile=RADIUS_PERCENTILE):
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    axis = vt[0]
    t = (pts - c) @ axis
    radial = np.linalg.norm((pts - c) - np.outer(t, axis), axis=1)
    r = float(np.percentile(radial, radius_percentile))
    tmin, tmax = float(t.min()), float(t.max())
    center = c + axis * (tmin + tmax) / 2
    length = max(0.0, (tmax - tmin) - 2 * r)  # threepp adds the two caps (2r) on top
    return center, frame_with_z(axis), r, length


def fit_box(pts):
    """Box aligned with the link frame (the palm is almost square, so a PCA
    direction would be a meaningless diagonal), trimmed of outlier points."""
    lo = np.percentile(pts, 100 - BOX_PERCENTILE, axis=0)
    hi = np.percentile(pts, BOX_PERCENTILE, axis=0)
    return (lo + hi) / 2, np.eye(3), hi - lo


def set_collision(link_el, center, R, geom_tag, geom_attrs):
    for col in link_el.findall("collision"):
        link_el.remove(col)
    col = ET.SubElement(link_el, "collision")
    rpy = matrix_to_rpy(R)
    ET.SubElement(col, "origin", {"xyz": " ".join(f"{v:.6f}" for v in center),
                                  "rpy": " ".join(f"{v:.6f}" for v in rpy)})
    geom = ET.SubElement(col, "geometry")
    ET.SubElement(geom, geom_tag, geom_attrs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    src, out = os.path.expanduser(args.src), os.path.expanduser(args.out)

    world = make_world()
    robot = ICub(world, src, base_position=BASE_POSITION)
    world.step(1e-4)  # update the meshes' poses; joints stay at ~0
    art = robot.art.articulation
    links, meshes = list(art.links), list(robot.meshes)

    def same_link(a, b):
        pa, pb, qa, qb = a.position, b.position, a.quaternion, b.quaternion
        return (abs(pa.x - pb.x) + abs(pa.y - pb.y) + abs(pa.z - pb.z) < 1e-9 and
                abs(qa.w - qb.w) + abs(qa.x - qb.x) + abs(qa.y - qb.y) + abs(qa.z - qb.z) < 1e-9)

    tree = ET.parse(src)
    link_els = {l.get("name"): l for l in tree.getroot().findall("link")}
    frames = urdf_link_frames(tree.getroot(), BASE_POSITION)

    for name in HAND_LINKS + list(ARM_CAPSULES):
        target = art.link(name)
        i = next(k for k, l in enumerate(links) if same_link(l, target))
        pts = points_in_link_frame(meshes[i], frames[name])
        if name == "r_hand":
            center, R, size = fit_box(pts)
            set_collision(link_els[name], center, R, "box", {"size": " ".join(f"{s:.6f}" for s in size)})
            print(f"{name:<20} box     size={np.round(size * 1000, 1)} mm")
        else:
            center, R, r, length = fit_capsule(pts, ARM_CAPSULES.get(name, RADIUS_PERCENTILE))
            set_collision(link_els[name], center, R, "cylinder", {"radius": f"{r:.6f}", "length": f"{length:.6f}"})
            print(f"{name:<20} capsule r={r * 1000:4.1f} mm  total length={(length + 2 * r) * 1000:4.1f} mm")

    tree.write(out, encoding="utf-8", xml_declaration=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
