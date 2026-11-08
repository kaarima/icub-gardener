"""iCub in PhysX: load the gardener URDF and control it by joint names.

The articulation has 85 joints, but only 30 can really move (see
scripts/make_gardener_urdf.py): torso (3), right arm (7), right hand (20).
The others are locked by near-zero limits and simply get a 0 target.

Units: radians for joints, metres for positions. The world is z-up.
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import numpy as np
import threepp as tp
from threepp.urdf import UrdfArticulation

from gardener.sim.builder import build_articulation

TORSO = ["torso_pitch", "torso_roll", "torso_yaw"]
RIGHT_ARM = [
    "r_shoulder_pitch", "r_shoulder_roll", "r_shoulder_yaw",
    "r_elbow",
    "r_wrist_prosup", "r_wrist_pitch", "r_wrist_yaw",
]
FINGERS = ["thumb", "index", "middle", "ring", "little"]
RIGHT_HAND = [f"r_hand_{f}_{k}_joint" for f in FINGERS for k in range(4)]
CONTROLLED = TORSO + RIGHT_ARM + RIGHT_HAND
# (stiffness N*m/rad, damping N*m*s/rad, torque limit N*m) for the 20 hand joints.
DEFAULT_FINGER_DRIVE = (20.0, 0.1, 0.3)
# Solver position iterations for a robot built with finger_drive. Compared on the can lift at a 2 ms
# timestep: 12 and 32 pass 2/8, 64 and 128 pass 8/8, and 64 is nearly twice as fast as 128.
DEFAULT_SOLVER_ITERATIONS = 64


def read_joint_limits(urdf_path: str) -> dict[str, tuple[float, float]]:
    """{joint name: (lower, upper)} for every joint that has a <limit>."""
    limits = {}
    for j in ET.parse(urdf_path).getroot().findall("joint"):
        lim = j.find("limit")
        if lim is not None and "lower" in lim.attrib and "upper" in lim.attrib:
            limits[j.get("name")] = (float(lim.get("lower")), float(lim.get("upper")))
    return limits


def make_world(timestep: float = 0.002, max_substeps: int = 20) -> tp.PhysxWorld:
    """A PhysX world with z-up gravity (same convention as the iCub URDF)."""
    return tp.PhysxWorld(gravity=tp.Vector3(0, 0, -9.81), fixed_timestep=timestep, max_substeps=max_substeps)


class ICub:
    """The gardener iCub: fixed base, PD position drives on every joint."""

    def __init__(self, world: tp.PhysxWorld, urdf_path: str, *,
                 base_position=(0.0, 0.0, 0.05),
                 stiffness: float = 400.0, damping: float = 40.0, max_force: float = 30.0,
                 density: float = 1200.0,
                 finger_drive: tuple[float, float, float] | None = None,
                 solver_iterations: int = DEFAULT_SOLVER_ITERATIONS):
        """finger_drive = (stiffness, damping, torque limit) gives the 20 hand joints their own
        drive. It builds the robot from colliders only (see builder.py), so no visual meshes."""
        self.urdf_path = os.path.expanduser(urdf_path)
        self._links_by_name = None
        if finger_drive is None:
            # world.load_articulation (instead of threepp.urdf.load_articulation)
            # because only it accepts drive_limits_are_forces: without it PhysX
            # treats max_force as an impulse per 2 ms substep, i.e. ~500x too
            # strong, and the fingers slam objects away.
            art, meshes, joint_names = world.load_articulation(
                self.urdf_path, fixed_base=True,
                base_position=tuple(float(x) for x in base_position),
                default_density=float(density), stiffness=float(stiffness),
                damping=float(damping), max_force=float(max_force),
                drive_limits_are_forces=True,
            )
        else:
            hand = set(RIGHT_HAND)
            others = (float(stiffness), float(damping), float(max_force))
            art, meshes, joint_names, self._links_by_name, self._meshes_by_name = build_articulation(
                world, self.urdf_path, base_position,
                lambda joint: tuple(map(float, finger_drive)) if joint in hand else others,
                solver_iterations)
        self.art = UrdfArticulation(art, meshes, joint_names, world)
        self.names = list(self.art.joint_names)
        self.index = {n: i for i, n in enumerate(self.names)}
        missing = [n for n in CONTROLLED if n not in self.index]
        if missing:
            raise RuntimeError(f"joints missing from {self.urdf_path}: {missing}")

        limits = read_joint_limits(self.urdf_path)
        self.lower = np.array([limits.get(n, (-np.inf, np.inf))[0] for n in self.names])
        self.upper = np.array([limits.get(n, (-np.inf, np.inf))[1] for n in self.names])
        # Start every target at 0, pushed inside the joint limits (e.g. the
        # elbow can't go below ~15 deg, so its "zero" target becomes 15 deg).
        self.targets = np.clip(np.zeros(len(self.names)), self.lower, self.upper)
        self.controlled_idx = np.array([self.index[n] for n in CONTROLLED])

    # ---- reading the state ----------------------------------------------
    def link(self, name: str):
        """The PhysX link for a URDF link name, whichever way the robot was built."""
        if self._links_by_name is not None:
            return self._links_by_name[name]
        return self.art.articulation.link(name)

    @property
    def body_names(self) -> list[str]:
        """Every body of a robot built with finger_drive."""
        return list(self._links_by_name)

    def collider(self, name: str):
        """The collider mesh of a URDF link; a ContactSensor attaches to this."""
        if self._links_by_name is not None:
            return self._meshes_by_name[name]
        target, links = self.link(name), list(self.art.articulation.links)
        pa, qa = target.position, target.quaternion
        for mesh, other in zip(self.art.meshes, links):
            pb, qb = other.position, other.quaternion
            if (abs(pa.x - pb.x) + abs(pa.y - pb.y) + abs(pa.z - pb.z) < 1e-9 and
                    abs(qa.w - qb.w) + abs(qa.x - qb.x) + abs(qa.y - qb.y) + abs(qa.z - qb.z) < 1e-9):
                return mesh
        raise KeyError(name)

    @property
    def num_dof(self) -> int:
        return len(self.names)

    @property
    def meshes(self):
        return self.art.meshes

    def positions(self) -> np.ndarray:
        return np.asarray(self.art.positions(), dtype=np.float64)

    def velocities(self) -> np.ndarray:
        return np.asarray(self.art.velocities(), dtype=np.float64)

    def get(self, names: list[str]) -> np.ndarray:
        """Current joint angles (rad) for the given joint names."""
        p = self.positions()
        return np.array([p[self.index[n]] for n in names])

    # ---- commanding -----------------------------------------------------
    def set_targets(self, targets: dict[str, float]) -> None:
        """Set target angles (rad) by joint name; values are clamped to the limits."""
        for name, value in targets.items():
            i = self.index[name]
            self.targets[i] = float(np.clip(value, self.lower[i], self.upper[i]))

    def hand_targets(self, closure: float) -> dict[str, float]:
        """Hand synergy: one number for the whole hand.

        closure = 0 -> open, 1 -> closed. Flexion joints (_1, _2, _3) go from
        their lower to their upper limit. The finger spread joints (_0) stay
        at 0, and the thumb's _0 joint (opposition) sits halfway so the thumb
        closes against the fingers.
        """
        s = float(np.clip(closure, 0.0, 1.0))
        out = {}
        for name in RIGHT_HAND:
            i = self.index[name]
            lo, hi = self.lower[i], self.upper[i]
            if name.endswith("_0_joint"):
                out[name] = lo + 0.5 * (hi - lo) if "thumb" in name else 0.0
            else:
                out[name] = lo + s * (hi - lo)
        return out

    def apply(self) -> None:
        """Send the current targets to the PhysX drives."""
        self.art.set_targets(self.targets.astype(np.float32))

    # ---- diagnostics ----------------------------------------------------
    def tracking_error(self) -> float:
        """Largest |target - position| over the 30 controlled joints (rad)."""
        p = self.positions()
        return float(np.max(np.abs(self.targets[self.controlled_idx] - p[self.controlled_idx])))

    def max_speed(self) -> float:
        """Largest |velocity| over the 30 controlled joints (rad/s)."""
        return float(np.max(np.abs(self.velocities()[self.controlled_idx])))
