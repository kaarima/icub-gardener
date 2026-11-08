"""Plan the reach and approach to the can's handle, using IK along straight lines.

This stands in for cuRobo until it is set up: it follows straight hand paths and checks the arm stays
above the table, but it does not avoid obstacles by searching.

Frames: the can stands upright and rotates about the centre of its body. iCub faces -x. The palm's
normal is the hand's -y axis, so the approach moves along -y of the hand towards the handle.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from gardener.sim.kinematics import ArmKinematics, rpy_matrix
from gardener.sim.robot import RIGHT_ARM, TORSO
from gardener.sim.scene import CanGeometry

WRIST_YAW = "r_wrist_yaw"
# Spare freedom is used to keep these joints away from their limits.
CENTER_WEIGHTS = {"r_wrist_yaw": 1.0, "r_wrist_pitch": 0.5, "r_wrist_prosup": 0.2,
                  "torso_pitch": 0.3, "torso_roll": 0.3, "torso_yaw": 0.3}
# The torso stays near upright if it can. wrist_yaw rests below mid-range: a negative angle keeps the
# forearm rising away from the hand, which is what holds it above the table.
REST = {"torso_pitch": 0.0, "torso_roll": 0.0, "torso_yaw": 0.0, "r_wrist_yaw": math.radians(-10.0)}
PLAN_JOINTS = TORSO + RIGHT_ARM


@dataclass
class Scene:
    """Where the nominal (0 degree) can stands, worked out from the grasp_table pose."""
    body_centre: np.ndarray        # x, y of the body centre, and z of the handle centre
    yaw0: float                    # direction from handle to body at 0 degrees
    table_z: float
    R0: np.ndarray                 # hand orientation for the 0 degree can
    handle_in_hand: np.ndarray


@dataclass
class Plan:
    can_yaw_deg: float
    T_grasp: np.ndarray
    q_ready: dict
    q_pre: dict
    q_grasp: dict
    path_reach: list = field(default_factory=list)     # ready -> pre-grasp
    path_approach: list = field(default_factory=list)  # pre-grasp -> grasp
    position_error: float = 0.0
    rotation_error: float = 0.0
    wrist_yaw_margin: float = 0.0  # smallest distance to a wrist_yaw limit on the whole path (rad)
    min_clearance: dict = field(default_factory=dict)  # smallest table clearance per link (m)
    hand_yaw_offset_deg: float = 0.0  # how far the hand's yaw differs from the can's


def nominal_scene(kin: ArmKinematics, q_nominal: dict, handle_in_hand, can: CanGeometry, yaw_shift_deg: float = 0.0) -> Scene:
    """yaw_shift_deg turns the nominal hand and can about the handle, to centre the arm's range."""
    T = kin.hand(q_nominal)
    handle = T[:3, :3] @ np.asarray(handle_in_hand) + T[:3, 3]
    T[:3, :3] = rpy_matrix(0.0, 0.0, math.radians(yaw_shift_deg)) @ T[:3, :3]
    u = -T[:3, 1].copy()
    u[2] = 0.0
    u /= np.linalg.norm(u)
    centre = handle + u * can.handle_to_body_centre
    return Scene(body_centre=centre, yaw0=math.atan2(u[1], u[0]), table_z=float(handle[2] - can.height / 2),
                 R0=T[:3, :3].copy(), handle_in_hand=np.asarray(handle_in_hand, dtype=float))


def handle_position(scene: Scene, can: CanGeometry, can_yaw: float) -> np.ndarray:
    """World position of the handle centre when the can is rotated by can_yaw about its body centre."""
    a = scene.yaw0 + can_yaw
    return scene.body_centre - np.array([math.cos(a), math.sin(a), 0.0]) * can.handle_to_body_centre


def plan_grasp(kin: ArmKinematics, scene: Scene, can: CanGeometry, can_yaw: float, q_seed: dict, table_back_x: float,
               pre_distance=0.07, ready_up=0.08, lift=0.03, path_points=12, restarts=12, seed=0,
               offsets_deg=(0, 5, -5, 10, -10, 15, -15, 20, -20, 25, -25, 30, -30), min_margin_deg=5.0,
               min_clearance=0.003, seat_offset_deg=0.0) -> Plan:
    """Choose how the hand meets the handle for this can rotation, and plan the whole path.

    The handle is a vertical bar, so the hand need not turn as far as the can. The hand's yaw offset
    from the can is the smallest in offsets_deg whose whole path (reach, approach, lift) keeps
    min_margin_deg of wrist_yaw and min_clearance above the table. If none does, the best one is
    returned, and the caller sees it in the plan's margin and clearance. seat_offset_deg is added to
    every offset: it is the angle at which the square handle settles flat in the closed hand.
    """
    rng = np.random.default_rng(seed)
    lo, hi = kin.limits(WRIST_YAW)
    handle = handle_position(scene, can, can_yaw)

    def solve(T_goal):
        best = None
        for k in range(restarts):
            start = dict(q_seed)
            if k:
                for n in PLAN_JOINTS:
                    a, b = kin.limits(n)
                    sigma = 0.1 if n in TORSO else 0.35
                    start[n] = float(np.clip(q_seed.get(n, 0.0) + rng.normal(0, sigma), a, b))
            q, p_err, r_err = kin.ik(start, T_goal, PLAN_JOINTS, CENTER_WEIGHTS, rest=REST)
            clear = min(kin.table_clearance(q, scene.table_z, table_back_x).values())
            margin = min(q[WRIST_YAW] - lo, hi - q[WRIST_YAW])
            # Once margin and clearance are good enough, more of either earns nothing.
            score = min(margin / math.radians(min_margin_deg), 1.0) + min(clear / 0.005, 1.0) - p_err * 100 - r_err
            if best is None or score > best[0]:
                best = (score, q, p_err, r_err)
        return best[1:]

    def walk(q_start, T_from, T_to):
        """Follow a straight hand line, each IK starting from the last; returns the joint path."""
        out, q, worst = [], dict(q_start), 0.0
        for s in np.linspace(0.0, 1.0, path_points + 1)[1:]:
            T = T_from.copy()
            T[:3, 3] = T_from[:3, 3] + s * (T_to[:3, 3] - T_from[:3, 3])
            q, p_err, r_err = kin.ik(q, T, PLAN_JOINTS, CENTER_WEIGHTS, iterations=25, rest=REST)
            worst = max(worst, p_err)
            out.append(dict(q))
        return out, worst

    best_plan, best_rank = None, None
    for offset in offsets_deg:
        hand_yaw = can_yaw + math.radians(seat_offset_deg + offset)
        R = rpy_matrix(0.0, 0.0, hand_yaw) @ scene.R0
        T_grasp = np.eye(4)
        T_grasp[:3, :3] = R
        T_grasp[:3, 3] = handle - R @ scene.handle_in_hand
        q_grasp, p_err, r_err = solve(T_grasp)
        if p_err > 0.002 or r_err > math.radians(2.0):
            continue

        T_pre = T_grasp.copy()
        T_pre[:3, 3] += R[:, 1] * pre_distance   # back off along the palm normal
        T_ready = T_pre.copy()
        T_ready[2, 3] += ready_up
        T_lift = T_grasp.copy()
        T_lift[2, 3] += lift

        # Walk backwards from the grasp so the path ends exactly on the chosen configuration.
        back_to_pre, e1 = walk(q_grasp, T_grasp, T_pre)
        back_to_ready, e2 = walk(back_to_pre[-1], T_pre, T_ready)
        lift_path, e3 = walk(q_grasp, T_grasp, T_lift)
        approach = back_to_pre[::-1][1:] + [q_grasp]
        reach = back_to_ready[::-1][1:] + [back_to_pre[-1]]

        whole = [back_to_ready[-1]] + reach + approach + lift_path
        margin = min(min(q[WRIST_YAW] - lo, hi - q[WRIST_YAW]) for q in whole)
        clearance = {}
        for q in whole:
            for name, c in kin.table_clearance(q, scene.table_z, table_back_x).items():
                clearance[name] = min(clearance.get(name, 1e9), c)
        path_err = max(e1, e2, e3)
        plan = Plan(can_yaw_deg=math.degrees(can_yaw), T_grasp=T_grasp, q_ready=back_to_ready[-1], q_pre=back_to_pre[-1],
                    q_grasp=q_grasp, path_reach=reach, path_approach=approach, position_error=max(p_err, path_err),
                    rotation_error=r_err, wrist_yaw_margin=margin, min_clearance=clearance, hand_yaw_offset_deg=offset)
        ok = (margin >= math.radians(min_margin_deg) and min(clearance.values()) >= min_clearance
              and plan.position_error < 0.002)
        if ok:
            return plan
        rank = min(margin / math.radians(min_margin_deg), min(clearance.values()) / min_clearance, 1.0) - plan.position_error * 100
        if best_rank is None or rank > best_rank:
            best_plan, best_rank = plan, rank
    return best_plan
