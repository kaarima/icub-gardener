"""Objects around iCub: the table and the watering can. All lengths in metres."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import threepp as tp

TABLE_SIZE = (0.60, 0.80, 0.03)  # x, y, thickness
TABLE_BACK_X = -0.11             # iCub faces -x; the chest ends at about x = -0.09


@dataclass
class CanGeometry:
    handle: float = 0.020          # handle width and depth
    handle_length: float = 0.110   # long enough that the connecting arms clear the fingers
    gap: float = 0.050             # between handle and body, room for the fingers
    body: float = 0.070            # body width and depth
    height: float = 0.125
    mass: float = 0.20
    top_arm: bool = True           # False: the handle joins the body at the bottom only, clear of the thumb

    @property
    def handle_to_body_centre(self) -> float:
        return self.handle / 2 + self.gap + self.body / 2


def box_mesh(size, pos, yaw=0.0):
    mesh = tp.Mesh(tp.BoxGeometry(*map(float, size)), tp.MeshStandardMaterial())
    mesh.position.set(*map(float, pos))
    mesh.quaternion.set(0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
    return mesh


def add_table(world, table_z: float, material):
    """A full-size table: its back edge stops just in front of the chest, and the arm reaches over it."""
    centre = (TABLE_BACK_X - TABLE_SIZE[0] / 2, 0.0, table_z - TABLE_SIZE[2] / 2)
    mesh = box_mesh(TABLE_SIZE, centre)
    world.add_static(mesh, material=material)
    return mesh


def build_can(world, handle_pos, u, table_z: float, can: CanGeometry, material, solver_iterations: int = 16):
    """Body on the far side of the handle along horizontal direction u. Returns (body link, meshes).

    One floating articulation of 4 boxes (body, handle, two connecting arms) joined by locked hinges:
    threepp's add() gives only one shape per body, and this version has no fixed joint.
    """
    handle_pos = np.asarray(handle_pos, dtype=float)
    yaw = math.atan2(u[1], u[0])
    arm_t, arm_mass, handle_mass = 0.010, 0.005, 0.010
    body_c = handle_pos + u * can.handle_to_body_centre
    body_c[2] = table_z + can.height / 2
    body_mass = can.mass - handle_mass - (2 if can.top_arm else 1) * arm_mass
    art = world.create_articulation(fixed_base=False, solver_position_iterations=max(16, solver_iterations),
                                    disable_self_collision=True, drive_limits_are_forces=True)
    body_mesh = box_mesh((can.body, can.body, can.height), body_c, yaw)
    body = art.add_link(body_mesh, density=body_mass / (can.body * can.body * can.height), material=material)
    meshes = {"body": body_mesh}

    def locked(name, size, pos, mass):
        mesh = box_mesh(size, pos, yaw)
        art.add_link(mesh, parent=body, density=mass / (size[0] * size[1] * size[2]), axis=[0.0, 0.0, 1.0],
                     anchor=[float(v) for v in pos], lower=-1e-5, upper=1e-5, material=material)
        meshes[name] = mesh

    locked("handle", (can.handle, can.handle, can.handle_length), handle_pos, handle_mass)
    for name, sign in ((("arm_top", 1), ("arm_bottom", -1)) if can.top_arm else (("arm_bottom", -1),)):
        p = handle_pos + u * (can.handle / 2 + can.gap / 2) + np.array([0.0, 0.0, sign * (can.handle_length / 2 - arm_t / 2)])
        locked(name, (can.gap + 0.002, can.handle, arm_t), p, arm_mass)
    art.finalize()
    return body, meshes
