"""The scripted gardener: one full episode from an instruction and a seed (book section 7.2).

reach -> approach -> wrap fingers -> curl thumb -> lift check -> transport -> pour -> transport back ->
place -> release -> retract. PhysX moves the robot and can; Warp moves the water, following the can.

The arm follows straight hand lines through IK. cuRobo can replace those plans later, as nothing here
depends on how a joint path was made.
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field, replace

import numpy as np
import yaml

from gardener.expert.planner import CENTER_WEIGHTS, PLAN_JOINTS, REST, WRIST_YAW, nominal_scene, plan_grasp
from gardener.sim.kinematics import ArmKinematics, quat_matrix, rpy_matrix
from gardener.sim.robot import DEFAULT_FINGER_DRIVE, ICub, make_world
from gardener.sim.scene import TABLE_BACK_X, CanGeometry, add_table, build_can
from gardener.sim.water import Pot, WaterParams, WaterSim, can_water_boxes, fill_lattice

BASE_POSITION = (0.0, 0.0, 0.05)
DT = 0.02
HANDLE_IN_HAND = (-0.070, -0.021, 0.0125)
YAW_SHIFT_DEG = 15.0        # centres the arm's range on the can's +-25 deg
SEAT_OFFSET_DEG = -3.0
THUMB_CURL = 0.7
CAN = CanGeometry(handle_length=0.085, height=0.100, top_arm=False)
INNER_HALF = 0.025          # the can's inside: 50 x 50 x 85 mm
INNER_HEIGHT = 0.085
FLOOR = 0.010
ABOVE_RIM = 0.03            # the can's floor above the pot's rim while pouring
AIM = 0.075                 # the can's origin stands this far back from the pot's centre


def parse_instruction(text: str) -> tuple[str, str]:
    """(pot, amount) from an instruction. Amount is 'little', 'lot' or 'normal'."""
    t = text.lower()
    pot = "left" if re.search(r"\bleft\b", t) else "right" if re.search(r"\bright\b", t) else None
    if pot is None:
        raise ValueError(f"no plant named in: {text!r}")
    return pot, "little" if "little" in t else "lot" if "a lot" in t else "normal"


@dataclass
class Result:
    instruction: str
    seed: int
    success: bool = False
    failure: str = ""
    stage_reached: str = ""
    poured_fraction: float = 0.0
    water_in_target: float = 0.0
    spill: float = 0.0
    wrong_pot: bool = False
    amount_ok: bool = False
    drag_mm: float = 0.0
    lift_mm: float = 0.0
    final_tilt_deg: float = 0.0
    placed_error_mm: float = 0.0
    min_wrist_yaw_margin_deg: float = 0.0
    sim_time: float = 0.0
    wall_time: float = 0.0
    counts: dict = field(default_factory=dict)
    log: list = field(default_factory=list)


def run_episode(urdf: str, config_path: str, poses_path: str, instruction: str, seed: int, verbose: bool = True,
                can_yaw_deg: float | None = None) -> Result:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    with open(poses_path) as f:
        pose = yaml.safe_load(f)["poses"]["grasp_table"]
    rng = np.random.default_rng(seed)
    r_cfg = cfg["randomise"]
    res = Result(instruction, seed)
    wall0 = time.perf_counter()

    def say(text):
        res.log.append(text)
        if verbose:
            print(text, flush=True)

    # ---- the scene for this seed ------------------------------------------------------------
    target_name, amount = parse_instruction(instruction)
    can_yaw = can_yaw_deg if can_yaw_deg is not None else rng.uniform(-r_cfg["can_yaw"], r_cfg["can_yaw"])
    fill = rng.uniform(*r_cfg["water_fill"])
    can_shift = rng.uniform(-r_cfg["can_position"], r_cfg["can_position"], 2)
    pot_shift = {k: rng.uniform(-r_cfg["pot_position"], r_cfg["pot_position"], 2) for k in ("left", "right")}
    n_water = cfg["water"]["particles"]
    inner_volume = (2 * INNER_HALF) ** 2 * INNER_HEIGHT
    spacing = (inner_volume * fill / n_water) ** (1 / 3)
    water_mass = inner_volume * fill * 1000.0
    can = replace(CAN, mass=cfg["can"]["empty_mass"] + water_mass)

    kin = ArmKinematics(urdf, base=BASE_POSITION)
    q_nominal = {n: math.radians(pose.get(n, 0.0)) for n in PLAN_JOINTS}
    scene = nominal_scene(kin, q_nominal, HANDLE_IN_HAND, can, YAW_SHIFT_DEG)
    scene.body_centre = scene.body_centre + np.array([can_shift[0], can_shift[1], 0.0])
    say(f"seed {seed}: {instruction!r}; can yaw {can_yaw:+.1f} deg, shift {np.round(can_shift * 1000)} mm, {fill * 100:.0f}% full, can {can.mass:.2f} kg")

    plan = plan_grasp(kin, scene, can, math.radians(can_yaw), q_nominal, TABLE_BACK_X, seat_offset_deg=SEAT_OFFSET_DEG)
    if plan is None or plan.position_error > 0.003:
        res.failure = "no plan"
        return res
    say(f"grasp plan: hand offset {plan.hand_yaw_offset_deg:+.0f} deg, wrist margin {math.degrees(plan.wrist_yaw_margin):.1f} deg, "
        f"clearance {min(plan.min_clearance.values()) * 1000:.1f} mm")
    lo, hi = kin.limits(WRIST_YAW)

    # ---- physics ------------------------------------------------------------------------------
    world = make_world()
    robot = ICub(world, urdf, base_position=BASE_POSITION, finger_drive=DEFAULT_FINGER_DRIVE)
    world.step(1e-4)
    state = dict(fingers=0.0, thumb=0.0, sim=0.0, margin=math.inf)

    def physics():
        robot.set_targets(robot.grasp_targets(state["fingers"], state["thumb"]))
        robot.apply()
        world.step(DT)
        state["sim"] += DT
        y = robot.get([WRIST_YAW])[0]
        state["margin"] = min(state["margin"], y - lo, hi - y)

    robot.set_targets({n: plan.q_ready[n] for n in PLAN_JOINTS})
    for _ in range(int(3.0 / DT)):
        physics()
    settled = dict(zip(PLAN_JOINTS, robot.get(PLAN_JOINTS)))
    sag = {n: robot.targets[robot.index[n]] - settled[n] for n in PLAN_JOINTS}

    material = world.create_material(0.5, 0.5, 0.0, friction_combine="min")
    add_table(world, scene.table_z, material)
    a = scene.yaw0 + math.radians(can_yaw)
    u = np.array([math.cos(a), math.sin(a), 0.0])
    handle_pos = scene.body_centre - u * can.handle_to_body_centre
    body, can_meshes = build_can(world, handle_pos, u, scene.table_z, can, material)
    pots = {}
    for k in ("left", "right"):
        p = cfg["pots"][k]
        pots[k] = Pot(x=p["x"] + pot_shift[k][0], y=p["y"] + pot_shift[k][1], z=scene.table_z)
    pot_list = [pots["left"], pots["right"]]
    target_index = 0 if target_name == "left" else 1

    def can_frame():
        """The can's water frame: origin on the inside floor, +x towards the spout."""
        p, q = body.position, body.quaternion
        T = np.eye(4)
        T[:3, :3] = quat_matrix(q.w, q.x, q.y, q.z)
        T[:3, 3] = np.array([p.x, p.y, p.z]) + T[:3, 2] * (FLOOR - can.height / 2)
        return T

    # ---- water --------------------------------------------------------------------------------
    T_can = can_frame()
    local = fill_lattice(INNER_HALF - spacing / 2, INNER_HALF - spacing / 2, INNER_HEIGHT - 0.002, spacing, n_water)
    table_box = ((TABLE_BACK_X - 0.30, 0.0, scene.table_z - 0.015), (0.30, 0.40, 0.015))
    water = WaterSim((T_can[:3, :3] @ local.T).T + T_can[:3, 3], pot_list, [table_box], WaterParams(spacing=spacing), ground=0.0)
    boxes = can_water_boxes(INNER_HALF, INNER_HALF, INNER_HEIGHT)

    def step():
        physics()
        water.set_can(can_frame(), boxes)
        water.step(DT)

    def counts():
        return water.count(can_frame(), INNER_HALF + 0.012, INNER_HALF + 0.012, INNER_HEIGHT + 0.005, scene.table_z)

    def body_tilt_deg():
        return math.degrees(math.acos(float(np.clip(can_frame()[2, 2], -1.0, 1.0))))

    def hold(seconds):
        for _ in range(int(seconds / DT)):
            step()

    def follow(path, seconds):
        steps = max(1, int(seconds / DT))
        start = {n: robot.targets[robot.index[n]] for n in PLAN_JOINTS}
        points = [start] + [{n: q[n] + sag[n] for n in PLAN_JOINTS} for q in path]
        for i in range(steps):
            s = (i + 1) / steps * (len(points) - 1)
            k = min(int(s), len(points) - 2)
            f = s - k
            robot.set_targets({n: points[k][n] + f * (points[k + 1][n] - points[k][n]) for n in PLAN_JOINTS})
            step()

    def measured_q():
        return dict(zip(PLAN_JOINTS, robot.get(PLAN_JOINTS)))

    def can_path(start_T, goal_T, q_start, points=16):
        """Joint path that carries the can in a straight line, turning it evenly, for this grip."""
        q, out, worst = dict(q_start), [], 0.0
        for s in np.linspace(0.0, 1.0, points + 1)[1:]:
            T = interpolate_pose(start_T, goal_T, s)
            q, e, r = kin.ik(q, T @ T_hand_in_can, PLAN_JOINTS, CENTER_WEIGHTS, iterations=30, rest=REST)
            worst = max(worst, e + r * 0.1)
            out.append(dict(q))
        return out, worst

    # ---- reach and grasp ----------------------------------------------------------------------
    hold(0.5)
    rest = can_frame()
    res.stage_reached = "reach"
    follow(plan.path_reach, 2.0)
    res.stage_reached = "approach"
    follow(plan.path_approach, 2.0)
    hold(0.3)
    res.stage_reached = "grasp"
    n = int(2.0 / DT)
    for i in range(n):
        state["fingers"] = (i + 1) / n
        step()
    hold(0.3)
    n = int(1.5 * THUMB_CURL / DT)
    for i in range(n):
        state["thumb"] = THUMB_CURL * (i + 1) / n
        step()
    hold(1.0)
    closed = can_frame()
    res.drag_mm = float(np.linalg.norm((closed - rest)[:2, 3])) * 1000
    say(f"grasped: drag {res.drag_mm:.1f} mm")

    # ---- lift check -----------------------------------------------------------------------------
    res.stage_reached = "lift"
    q = measured_q()
    T_hand = kin.hand(q)
    T_hand_in_can = np.linalg.inv(closed) @ T_hand
    lift_goal = closed.copy()
    lift_goal[2, 3] += 0.03
    path, _ = can_path(closed, lift_goal, q)
    follow(path, 1.5)
    hold(0.5)
    now = can_frame()
    res.lift_mm = float(now[2, 3] - closed[2, 3]) * 1000
    say(f"lift check: can rose {res.lift_mm:.1f} mm, tilt {body_tilt_deg():.1f} deg")
    if res.lift_mm < 15.0:
        res.failure = "grasp failure: can did not lift"
        return finish(res, state, wall0, counts())
    # Measure the grip again now that the can hangs in the hand.
    q = measured_q()
    T_hand_in_can = np.linalg.inv(now) @ kin.hand(q)

    # ---- transport to the pour pose ---------------------------------------------------------------
    res.stage_reached = "transport"
    pot = pots[target_name]
    rim = pot.z + pot.height
    heading = best_heading(pot, now)
    pour_origin = np.array([pot.x, pot.y, rim + ABOVE_RIM]) - np.array([math.cos(heading), math.sin(heading), 0.0]) * AIM
    pre_pour = np.eye(4)
    pre_pour[:3, :3] = rpy_matrix(0.0, 0.0, heading)
    pre_pour[:3, 3] = pour_origin
    above = now.copy()
    above[2, 3] = pour_origin[2]
    path1, e1 = can_path(now, above, q)
    path2, e2 = can_path(above, pre_pour, path1[-1], points=24)
    say(f"transport plan: heading {math.degrees(heading):.0f} deg, error {max(e1, e2) * 1000:.1f} mm")
    if max(e1, e2) > 0.01:
        res.failure = "no transport plan"
        return finish(res, state, wall0, counts())
    follow(path1, 1.2)
    follow(path2, 2.5)
    hold(0.5)
    if can_frame()[2, 3] < rim:
        res.failure = "grasp failure: dropped during transport"
        return finish(res, state, wall0, counts())

    # ---- pour: tilt until the requested amount has left, using the particle count -----------------
    res.stage_reached = "pour"
    band = {"little": cfg["expert"]["little"], "lot": cfg["expert"]["a_lot"], "normal": [0.40, 0.60]}[amount]
    goal_fraction = sum(band) / 2
    n_total = water.n
    can_now = can_frame()
    q = measured_q()
    T_hand_in_can = np.linalg.inv(can_now) @ kin.hand(q)
    rate = math.radians(cfg["expert"]["tilt_rate"])
    max_tilt = math.radians(cfg["expert"]["max_tilt"])
    tilt, poured = 0.0, 0.0
    q_tilt = dict(q)
    hold_time = 0.0
    while tilt < max_tilt or hold_time < 3.0:
        poured = (n_total - counts()["can"]) / n_total
        # Allow for the water still in the air: stop tilting a little before the goal.
        if poured >= goal_fraction - 0.05:
            break
        if tilt < max_tilt:
            # Pause while water is flowing, so the amount can be judged.
            flowing = counts()["air"] > 0.01 * n_total
            tilt = min(max_tilt, tilt + rate * DT * (0.25 if flowing else 1.0))
        else:
            hold_time += DT
        T = pour_pose(can_now, tilt)
        q_tilt, _, _ = kin.ik(q_tilt, T @ T_hand_in_can, PLAN_JOINTS, CENTER_WEIGHTS, iterations=8, rest=REST)
        robot.set_targets({n: q_tilt[n] + sag[n] for n in PLAN_JOINTS})
        step()
        if can_frame()[2, 3] < rim - 0.05:
            res.failure = "grasp failure: dropped while pouring"
            return finish(res, state, wall0, counts())
    say(f"pour: stopped at tilt {math.degrees(tilt):.0f} deg with {poured * 100:.0f}% out")
    res.stage_reached = "tilt back"
    steps = int(max(1.0, tilt / rate) / DT)
    for i in range(steps):
        T = pour_pose(can_now, tilt * (1 - (i + 1) / steps))
        q_tilt, _, _ = kin.ik(q_tilt, T @ T_hand_in_can, PLAN_JOINTS, CENTER_WEIGHTS, iterations=8, rest=REST)
        robot.set_targets({n: q_tilt[n] + sag[n] for n in PLAN_JOINTS})
        step()
    hold(1.0)

    # ---- back to the table -----------------------------------------------------------------------
    res.stage_reached = "return"
    now = can_frame()
    q = measured_q()
    T_hand_in_can = np.linalg.inv(now) @ kin.hand(q)
    over = closed.copy()
    over[2, 3] = now[2, 3]
    place = closed.copy()
    place[2, 3] = rest[2, 3] + 0.004
    path1, e1 = can_path(now, over, q, points=24)
    path2, e2 = can_path(over, place, path1[-1])
    if max(e1, e2) > 0.01:
        res.failure = "no return plan"
        return finish(res, state, wall0, counts())
    follow(path1, 2.5)
    follow(path2, 1.5)
    hold(0.5)

    # ---- release and retract -----------------------------------------------------------------------
    res.stage_reached = "release"
    n = int(1.0 / DT)
    for i in range(n):
        state["thumb"] = THUMB_CURL * (1 - (i + 1) / n)
        step()
    n = int(1.5 / DT)
    for i in range(n):
        state["fingers"] = 1 - (i + 1) / n
        step()
    res.stage_reached = "retract"
    q = measured_q()
    T = kin.hand(q)
    back = T.copy()
    back[:3, 3] += T[:3, 1] * 0.07
    up = back.copy()
    up[2, 3] += 0.06
    path = []
    for start_T, goal_T in ((T, back), (back, up)):
        for s in np.linspace(0.0, 1.0, 11)[1:]:
            q, _, _ = kin.ik(q, interpolate_pose(start_T, goal_T, s), PLAN_JOINTS, CENTER_WEIGHTS, iterations=25, rest=REST)
            path.append(dict(q))
    follow(path, 2.0)
    hold(1.0)
    res.stage_reached = "end"

    # ---- score: the book's definition, fixed before evaluating any policy ----------------------------
    end = can_frame()
    c = counts()
    res.final_tilt_deg = body_tilt_deg()
    res.placed_error_mm = float(np.linalg.norm((end - rest)[:2, 3])) * 1000
    left = n_total - c["can"]
    res.poured_fraction = left / n_total
    res.water_in_target = c[f"pot{target_index}"] / max(left, 1)
    res.spill = (c.get("table", 0) + c.get("floor", 0)) / max(left, 1)
    res.wrong_pot = c[f"pot{1 - target_index}"] > c[f"pot{target_index}"]
    res.amount_ok = band[0] <= res.poured_fraction <= band[1]
    s = cfg["success"]
    on_table = abs(end[2, 3] - rest[2, 3]) < 0.01 and res.final_tilt_deg <= s["max_can_tilt"]
    res.success = (left > 0 and res.water_in_target >= s["min_water_in_target"] and on_table and state["sim"] <= s["timeout"])
    if not res.success:
        res.failure = ("nothing poured" if left == 0 else "wrong pot" if res.wrong_pot else "water missed the pot"
                       if res.water_in_target < s["min_water_in_target"] else "can not upright on the table" if not on_table else "timeout")
    return finish(res, state, wall0, c)


def finish(res: Result, state: dict, wall0: float, c: dict) -> Result:
    res.sim_time = state["sim"]
    res.wall_time = time.perf_counter() - wall0
    res.min_wrist_yaw_margin_deg = math.degrees(state["margin"])
    res.counts = c
    return res


def interpolate_pose(A: np.ndarray, B: np.ndarray, s: float) -> np.ndarray:
    """A straight line in position, and an even turn about one axis in orientation."""
    T = np.eye(4)
    T[:3, 3] = A[:3, 3] + s * (B[:3, 3] - A[:3, 3])
    dR = B[:3, :3] @ A[:3, :3].T
    angle = math.acos(float(np.clip((np.trace(dR) - 1) / 2, -1.0, 1.0)))
    if angle < 1e-6:
        T[:3, :3] = A[:3, :3]
        return T
    axis = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) / (2 * math.sin(angle))
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    t = s * angle
    T[:3, :3] = (np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * K @ K) @ A[:3, :3]
    return T


def pour_pose(can_upright: np.ndarray, tilt: float) -> np.ndarray:
    """The can tilted about its own y axis, spout side down, about the origin on its floor."""
    c, s = math.cos(tilt), math.sin(tilt)
    Ry = np.eye(4)
    Ry[:3, :3] = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return can_upright @ Ry


def best_heading(pot: Pot, can_T: np.ndarray) -> float:
    """Spout direction for pouring: the pot position search found these work for each side of iCub."""
    if pot.y > 0.15:
        return math.radians(-120.0)
    if pot.y < 0.02:
        return math.radians(-60.0)
    return math.radians(-90.0)
