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
import threepp as tp
import yaml

from gardener.expert.planner import CENTER_WEIGHTS, PLAN_JOINTS, REST, WRIST_YAW, nominal_scene, plan_grasp
from gardener.sim.kinematics import ArmKinematics, quat_matrix, rpy_matrix
from gardener.sim.robot import DEFAULT_FINGER_DRIVE, FINGERS, ICub, make_world
from gardener.sim.scene import TABLE_BACK_X, TABLE_SIZE, CanGeometry, add_pot, add_table, build_can, can_clear_of_pot
from gardener.sim.water import Pot, WaterParams, WaterSim, can_water_boxes, fill_lattice

BASE_POSITION = (0.0, 0.0, 0.05)
DT = 0.02
HANDLE_IN_HAND = (-0.070, -0.021, 0.0125)
YAW_SHIFT_DEG = 15.0        # centres the arm's range on the can's +-25 deg
SEAT_OFFSET_DEG = -3.0
THUMB_CURL = 0.85
CAN = CanGeometry(handle_length=0.085, height=0.100, top_arm=False)
INNER_HALF = 0.025          # the can's inside: 50 x 50 x 85 mm
INNER_HEIGHT = 0.085
SPOUT_DROP = 0.025          # how far the spout notch sits below the walls' top
USABLE_HEIGHT = INNER_HEIGHT - SPOUT_DROP   # water above this would run out of the notch at rest
FLOOR = 0.010
SPOUT = np.array([0.030, 0.0, USABLE_HEIGHT])  # the notch in the can's frame: the pivot for pouring
CARRY_ABOVE_RIM = 0.040     # the can's floor above the rims while crossing the table, before descending to pour
SPOUT_ABOVE_RIM = 0.075     # the upright can's floor then clears the rim by 15 mm
# Where the open hand reaches, in the hand frame (m): fingertips, finger middles, palm corners and the opposed thumb.
OPEN_HAND_POINTS = np.array([[-0.129, 0.005, 0.064], [-0.138, 0.006, 0.043], [-0.140, 0.012, 0.019], [-0.135, 0.011, 0.002],
                             [-0.100, 0.006, 0.050], [-0.100, 0.006, 0.010], [-0.068, 0.0, 0.047], [-0.068, 0.0, -0.022],
                             [0.013, 0.0, 0.047], [0.013, 0.0, -0.022], [-0.051, -0.066, 0.077], [-0.042, -0.035, 0.055]])
OPEN_HAND_RADIUS = 0.010
FLOW_LEAD_S = 0.5           # s of the current flow still to come after the can starts tilting back
TILT_BACK_RATE = 45.0       # deg/s; quick, so the flow stops soon
MIN_POUR_MARGIN_DEG = 5.0   # wrist_yaw must keep this much room all through the pour
MIN_SCENERY_FORCE = 10.0    # N. Sensors cannot tell a light pot contact from the fingers touching the can body
HAND_PARTS = ["r_hand"] + [f"r_hand_{f}_{k}" for f in FINGERS for k in range(4)]


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
    layouts_redrawn: int = 0
    spilled_before_pour: float = 0.0
    spout_error_mm: float = 0.0
    contacts: list = field(default_factory=list)  # things that should never touch, see run_episode
    contact_states: dict = field(default_factory=dict)  # diagnostics: the arm's state at the strongest moment of each contact
    counts: dict = field(default_factory=dict)
    log: list = field(default_factory=list)


def run_episode(urdf: str, config_path: str, poses_path: str, instruction: str, seed: int, verbose: bool = True,
                can_yaw_deg: float | None = None, recorder=None) -> Result:
    """recorder: an optional gardener.data.record.EpisodeRecorder. It only reads the simulation."""
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
    n_water = cfg["water"]["particles"]
    inner_volume = (2 * INNER_HALF) ** 2 * USABLE_HEIGHT   # "90% full" means 90% of what the can holds upright
    spacing = (inner_volume * fill / n_water) ** (1 / 3)
    water_mass = inner_volume * fill * 1000.0
    can = replace(CAN, mass=cfg["can"]["empty_mass"] + water_mass)

    kin = ArmKinematics(urdf, base=BASE_POSITION)
    q_nominal = {n: math.radians(pose.get(n, 0.0)) for n in PLAN_JOINTS}
    scene = nominal_scene(kin, q_nominal, HANDLE_IN_HAND, can, YAW_SHIFT_DEG)
    nominal_centre = scene.body_centre.copy()
    pot_outer = Pot(0, 0, 0).radius + Pot(0, 0, 0).wall
    rim_z = scene.table_z + Pot(0, 0, 0).height
    # Redraw positions until nothing overlaps: first the cheap footprint checks, then the planned arm.
    plan = None
    for _ in range(200):
        can_shift = rng.uniform(-r_cfg["can_position"], r_cfg["can_position"], 2)
        pot_shift = {k: rng.uniform(-r_cfg["pot_position"], r_cfg["pot_position"], 2) for k in ("left", "right")}
        scene.body_centre = nominal_centre + np.array([can_shift[0], can_shift[1], 0.0])
        a_ = scene.yaw0 + math.radians(can_yaw)
        h = scene.body_centre - np.array([math.cos(a_), math.sin(a_), 0.0]) * can.handle_to_body_centre
        pot_xy = {k: (cfg["pots"][k]["x"] + pot_shift[k][0], cfg["pots"][k]["y"] + pot_shift[k][1]) for k in ("left", "right")}
        if (all(can_clear_of_pot(scene.body_centre, h, pot_xy[k], pot_outer) for k in pot_xy)
                and all(open_hand_clear(scene, can, can_yaw, pot_xy[k], pot_outer) for k in pot_xy)):
            candidate = plan_grasp(kin, scene, can, math.radians(can_yaw), q_nominal, TABLE_BACK_X, seat_offset_deg=SEAT_OFFSET_DEG)
            if candidate is not None and candidate.position_error <= 0.004 and arm_clear_of_pots(kin, candidate, list(pot_xy.values()), pot_outer, rim_z):
                plan = candidate
                break
        res.layouts_redrawn += 1
    say(f"seed {seed}: {instruction!r}; can yaw {can_yaw:+.1f} deg, shift {np.round(can_shift * 1000)} mm, {fill * 100:.0f}% full, can {can.mass:.2f} kg")
    if plan is None:
        res.failure = "no plan"
        return res
    say(f"grasp plan: hand offset {plan.hand_yaw_offset_deg:+.0f} deg, wrist margin {math.degrees(plan.wrist_yaw_margin):.1f} deg, "
        f"clearance {min(plan.min_clearance.values()) * 1000:.1f} mm")
    lo, hi = kin.limits(WRIST_YAW)

    # ---- physics ------------------------------------------------------------------------------
    world = make_world()
    robot = ICub(world, urdf, base_position=BASE_POSITION, finger_drive=DEFAULT_FINGER_DRIVE)
    world.step(1e-4)
    state = dict(fingers=0.0, thumb=0.0, sim=0.0, margin=math.inf, phase="setup", can_in_air=False)
    other_bodies = [n for n in robot.body_names if n not in HAND_PARTS]
    sensors = {n: tp.ContactSensor(robot.collider(n)) for n in robot.body_names}
    for sensor in sensors.values():
        world.register_sensor(sensor)
    can_body_sensor = [None]
    can_meshes_ref = [None]
    table_z_ref = [scene.table_z]
    pot_list = []
    contacts = {}
    contact_states = res.contact_states

    def can_meshes_pose():
        """World pose (position, quaternion wxyz) of each part of the can, for diagnostics."""
        if can_meshes_ref[0] is None:
            return {}
        out = {}
        for name, mesh in can_meshes_ref[0].items():
            p_, q_ = mesh.position, mesh.quaternion
            out[name] = ([p_.x, p_.y, p_.z], [q_.w, q_.x, q_.y, q_.z])
        return out

    def strongest(sensor):
        force = 0.0
        for sample in sensor.drain():
            f = sample.force
            force = max(force, math.sqrt(f.x * f.x + f.y * f.y + f.z * f.z))
        return force

    def note(what, force):
        if force > 0.0:
            key = f"{what} during {state['phase']}"
            if force > contacts.get(key, 0.0):
                contact_states[key] = dict(force=force, q={n: float(v) for n, v in zip(PLAN_JOINTS, robot.get(PLAN_JOINTS))},
                                           can=can_meshes_pose(), table_z=table_z_ref[0], pots=[(p_.x, p_.y, p_.z, p_.radius, p_.wall, p_.height) for p_ in pot_list])
            contacts[key] = max(contacts.get(key, 0.0), force)

    def contact_target(p):
        """What is at this contact point? Used to report what the arm hit."""
        if can_body_sensor[0] is not None:
            for mesh_name, half in (("body", (can.body / 2, can.body / 2, can.height / 2)),):
                c = can_meshes_ref[0][mesh_name].position
                q = can_meshes_ref[0][mesh_name].quaternion
                R = quat_matrix(q.w, q.x, q.y, q.z)
                local = R.T @ np.array([p.x - c.x, p.y - c.y, p.z - c.z])
                if all(abs(local[i]) < half[i] + 0.008 for i in range(3)):
                    return "can body"
        if touches_scenery(p, pot_list, table_z_ref[0]):
            return "pot/table"
        return "can handle or elsewhere"

    def physics():
        robot.set_targets(robot.grasp_targets(state["fingers"], state["thumb"]))
        robot.apply()
        world.step(DT)
        state["sim"] += DT
        for name, sensor in sensors.items():
            if name in other_bodies:                   # the arm and body must never touch anything
                for sample in sensor.drain():
                    f = sample.force
                    force = math.sqrt(f.x * f.x + f.y * f.y + f.z * f.z)
                    if force > 0.0:
                        target = "?"
                        for point in sample.points:
                            target = contact_target(point.position)
                        note(f"{name} on {target}", force)
            else:
                force = strongest(sensor)
                if state["phase"] in ("reach", "approach"):
                    note(name, force)                  # nor the hand, before the grasp starts
        if can_body_sensor[0] is not None:
            for sample in can_body_sensor[0].drain():
                if state["can_in_air"]:
                    for point in sample.points:        # fingers may touch the body; pots and the table may not
                        if touches_scenery(point.position, pot_list, scene.table_z):
                            force_here = point.impulse / (DT / 10)   # impulse over one 2 ms substep
                            if force_here > MIN_SCENERY_FORCE:
                                note("can body on a pot or the table", force_here)
        y = robot.get([WRIST_YAW])[0]
        state["margin"] = min(state["margin"], y - lo, hi - y)

    robot.set_targets({n: plan.q_ready[n] for n in PLAN_JOINTS})
    for _ in range(int(3.0 / DT)):
        physics()
    settled = dict(zip(PLAN_JOINTS, robot.get(PLAN_JOINTS)))
    sag = {n: robot.targets[robot.index[n]] - settled[n] for n in PLAN_JOINTS}

    material = world.create_material(0.5, 0.5, 0.0, friction_combine="min")
    table_mesh = add_table(world, scene.table_z, material)
    a = scene.yaw0 + math.radians(can_yaw)
    u = np.array([math.cos(a), math.sin(a), 0.0])
    handle_pos = scene.body_centre - u * can.handle_to_body_centre
    body, can_meshes = build_can(world, handle_pos, u, scene.table_z, can, material)
    pots = {}
    for k in ("left", "right"):
        p = cfg["pots"][k]
        pots[k] = Pot(x=p["x"] + pot_shift[k][0], y=p["y"] + pot_shift[k][1], z=scene.table_z)
    pot_meshes = []
    for q in pots.values():
        pot_meshes += add_pot(world, q.x, q.y, q.z, q.radius, q.height, q.wall, q.floor, material)
    can_meshes_ref[0] = can_meshes
    can_body_sensor[0] = tp.ContactSensor(can_meshes["body"])
    world.register_sensor(can_body_sensor[0])
    pot_list.extend([pots["left"], pots["right"]])
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
    local = fill_lattice(INNER_HALF - spacing / 2, INNER_HALF - spacing / 2, USABLE_HEIGHT, spacing, n_water)
    table_box = ((TABLE_BACK_X - TABLE_SIZE[0] / 2, 0.0, scene.table_z - TABLE_SIZE[2] / 2), tuple(v / 2 for v in TABLE_SIZE))
    water = WaterSim((T_can[:3, :3] @ local.T).T + T_can[:3, 3], pot_list, [table_box], WaterParams(spacing=spacing), ground=0.0)
    boxes = can_water_boxes(INNER_HALF, INNER_HALF, INNER_HEIGHT, spout_drop=SPOUT_DROP)
    if recorder is not None:
        recorder.add_robot(robot, HAND_PARTS)
        recorder.add_scenery(table_mesh, pot_meshes, list(can_meshes.values()), water.n, scene.table_z)

    def step():
        physics()
        water.set_can(can_frame(), boxes)
        water.step(DT)
        if recorder is not None and state["phase"] != "setup":
            q_now = dict(zip(PLAN_JOINTS, robot.get(PLAN_JOINTS)))
            recorder.frame(kin.fk("r_eye", q_now)[:3, 3], q_now, state["fingers"], state["thumb"],
                           {n: float(robot.targets[robot.index[n]]) for n in PLAN_JOINTS}, water.positions())

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
    res.stage_reached = state["phase"] = "reach"
    follow(plan.path_reach, 2.0)
    res.stage_reached = state["phase"] = "approach"
    follow(plan.path_approach, 2.0)
    hold(0.3)
    res.stage_reached = state["phase"] = "grasp"
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
    res.stage_reached = state["phase"] = "lift"
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
        return finish(res, state, wall0, counts(), contacts)
    state["can_in_air"] = True
    # Measure the grip again now that the can hangs in the hand.
    q = measured_q()
    T_hand_in_can = np.linalg.inv(now) @ kin.hand(q)

    # ---- transport to the pour pose ---------------------------------------------------------------
    res.stage_reached = state["phase"] = "transport"
    pot = pots[target_name]
    rim = pot.z + pot.height
    res.spilled_before_pour = 0.0
    all_pots = [(q_.x, q_.y) for q_ in pot_list]
    candidates = pour_headings(kin, pot, all_pots, T_hand_in_can, q, math.atan2(now[1, 0], now[0, 0]), pot_outer, rim, can)
    picked = None
    for furthest_tilt, _, _, heading in candidates[:8]:
        if furthest_tilt < 60:
            break
        pre_pour = upright_over_pot(pot, heading)
        # Up first, across high, then down: the can and arm cross above the pots' rims.
        above = now.copy()
        above[2, 3] = rim + CARRY_ABOVE_RIM
        high_over_pot = pre_pour.copy()
        high_over_pot[2, 3] = above[2, 3]
        path1, e1 = can_path(now, above, q)
        path2, e2 = can_path(above, high_over_pot, path1[-1], points=24)
        path_down, e3 = can_path(high_over_pot, pre_pour, path2[-1], points=6)
        whole = path1 + path2 + path_down
        if (max(e1, e2, e3) <= 0.01 and all(arm_clear_at(kin, q_, all_pots, pot_outer, rim) for q_ in whole)
                and all(arm_clear_of_can(kin, q_, T_hand_in_can, can) for q_ in whole)):
            picked = heading
            break
    if picked is None:
        res.failure = "no transport plan"
        return finish(res, state, wall0, counts(), contacts)
    say(f"transport plan: heading {math.degrees(heading):.0f} deg, tilt available {furthest_tilt} deg, error {max(e1, e2, e3) * 1000:.1f} mm")
    follow(path1, 1.8)
    follow(path2, 3.2)
    follow(path_down, 0.6)
    hold(0.5)
    # The can may have turned a little in the hand on the way: measure the grip and move the spout onto the target.
    for _ in range(2):
        arrived = can_frame()
        q = measured_q()
        T_hand_in_can = np.linalg.inv(arrived) @ kin.hand(q)
        path3, _ = can_path(arrived, pre_pour, q, points=6)
        follow(path3, 0.8)
        hold(0.3)
    arrived = can_frame()
    res.spout_error_mm = float(np.linalg.norm(((arrived - pre_pour) @ np.append(SPOUT, 1.0))[:2])) * 1000
    res.spilled_before_pour = (water.n - counts()["can"]) / water.n
    say(f"over the pot: spout {res.spout_error_mm:.1f} mm off target, {res.spilled_before_pour * 100:.0f}% of the water already lost")
    if can_frame()[2, 3] < rim:
        res.failure = "grasp failure: dropped during transport"
        return finish(res, state, wall0, counts(), contacts)

    # ---- pour: tilt until the requested amount has left, using the particle count -----------------
    res.stage_reached = state["phase"] = "pour"
    band = {"little": cfg["expert"]["little"], "lot": cfg["expert"]["a_lot"], "normal": [0.40, 0.60]}[amount]
    goal_fraction = sum(band) / 2
    n_total = water.n
    can_now = pre_pour                # tilt about the target, with the grip as measured
    q = measured_q()
    T_hand_in_can = np.linalg.inv(can_frame()) @ kin.hand(q)
    rate = math.radians(cfg["expert"]["tilt_rate"])
    max_tilt = math.radians(min(cfg["expert"]["max_tilt"], furthest_tilt))
    tilt, poured = 0.0, 0.0
    spout = can_now @ np.append(SPOUT, 1.0)
    say(f"pour start: spout {np.round((spout[:3] - [pot.x, pot.y, rim]) * 1000)} mm from the pot's rim centre, "
        f"can up axis {np.round(can_now[:3, 2], 2)}, spout direction {np.round(can_now[:3, 0], 2)}, planned heading {math.degrees(heading):.0f} deg")
    q_tilt = dict(q)
    hold_time = 0.0
    history = []
    while tilt < max_tilt or hold_time < 3.0:
        poured = (n_total - counts()["can"]) / n_total
        history.append(poured)
        # Water keeps coming for a moment after the can starts tilting back, so stop early by the
        # amount the current flow would still deliver.
        window = int(0.5 / DT)
        flow = (history[-1] - history[-window]) / (window * DT) if len(history) >= window else 0.0
        if poured + flow * FLOW_LEAD_S >= goal_fraction:
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
            return finish(res, state, wall0, counts(), contacts)
    spout = can_frame() @ np.append(SPOUT, 1.0)
    say(f"pour: stopped at tilt {math.degrees(tilt):.0f} deg with {poured * 100:.0f}% out; at the stop the spout is "
        f"{np.round((spout[:3] - [pot.x, pot.y, rim]) * 1000)} mm from the pot's rim centre, can up axis {np.round(can_frame()[:3, 2], 2)}")
    res.stage_reached = state["phase"] = "tilt back"
    steps = int(max(0.5, tilt / math.radians(TILT_BACK_RATE)) / DT)
    for i in range(steps):
        T = pour_pose(can_now, tilt * (1 - (i + 1) / steps))
        q_tilt, _, _ = kin.ik(q_tilt, T @ T_hand_in_can, PLAN_JOINTS, CENTER_WEIGHTS, iterations=8, rest=REST)
        robot.set_targets({n: q_tilt[n] + sag[n] for n in PLAN_JOINTS})
        step()
    hold(1.0)

    # ---- back to the table -----------------------------------------------------------------------
    res.stage_reached = state["phase"] = "return"
    now = can_frame()
    q = measured_q()
    T_hand_in_can = np.linalg.inv(now) @ kin.hand(q)
    # Retrace the transport's joint path: it is known to be reachable and clear of the pots.
    follow(path_down[::-1], 0.6)
    follow(path2[::-1], 3.2)
    follow(path1[::-1], 1.8)
    hold(0.3)
    now = can_frame()
    q = measured_q()
    T_hand_in_can = np.linalg.inv(now) @ kin.hand(q)
    state["can_in_air"] = False   # it is about to reach the table
    place = now.copy()
    place[:2, 3] = rest[:2, 3]
    place[:3, :3] = rest[:3, :3] @ (np.linalg.inv(closed[:3, :3]) @ closed[:3, :3])
    place[2, 3] = rest[2, 3] + 0.004
    path_down, e_down = can_path(now, place, q)
    say(f"return: back over the start, {np.linalg.norm((now - lift_goal)[:2, 3]) * 1000:.1f} mm from the lift point, lowering error {e_down * 1000:.1f} mm")
    if e_down > 0.015:
        res.failure = "no place plan"
        return finish(res, state, wall0, counts(), contacts)
    follow(path_down, 1.5)
    hold(0.5)

    # ---- release and retract -----------------------------------------------------------------------
    res.stage_reached = state["phase"] = "release"
    n = int(1.0 / DT)
    for i in range(n):
        state["thumb"] = THUMB_CURL * (1 - (i + 1) / n)
        step()
    n = int(1.5 / DT)
    for i in range(n):
        state["fingers"] = 1 - (i + 1) / n
        step()
    res.stage_reached = state["phase"] = "retract"
    q = measured_q()
    T = kin.hand(q)
    # Straight up first, along the handle: backing away at table height drags the forearm on the table.
    up = T.copy()
    up[2, 3] += 0.06
    back = up.copy()
    back[:3, 3] += T[:3, 1] * 0.07
    path = []
    for start_T, goal_T in ((T, up), (up, back)):
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
    res.success = (left > 0 and res.water_in_target >= s["min_water_in_target"] and on_table and state["sim"] <= s["timeout"]
                   and not contacts)
    if not res.success:
        res.failure = ("nothing poured" if left == 0 else "wrong pot" if res.wrong_pot else "water missed the pot"
                       if res.water_in_target < s["min_water_in_target"] else "can not upright on the table" if not on_table
                       else "timeout" if state["sim"] > s["timeout"] else "collision")
    return finish(res, state, wall0, c, contacts)


def arm_clear_at(kin, q: dict, pots_xy: list, pot_outer: float, rim_z: float, margin: float = 0.010,
                 links=("r_upper_arm", "r_elbow_1", "r_forearm", "r_wrist_1", "r_hand")) -> bool:
    """False if any of these arm colliders would meet a pot in this configuration."""
    points = kin._points()
    for name in links:
        pts, radius = points[name]
        T = kin.fk(name, q)
        w = (T[:3, :3] @ pts.T).T + T[:3, 3]
        low = w[w[:, 2] - radius < rim_z + margin]
        for x, y in pots_xy:
            if len(low) and np.min(np.hypot(low[:, 0] - x, low[:, 1] - y)) < pot_outer + radius + margin:
                return False
    return True


def can_boxes_in_can_frame(can: CanGeometry) -> list:
    """(centre, half sizes) of the can's solid parts in its water frame (origin on the inside floor, +x to the spout)."""
    z0 = can.height / 2 - FLOOR
    handle_x = -can.handle_to_body_centre
    arm_z = z0 - can.handle_length / 2 + 0.005
    return [(np.array([0.0, 0.0, z0]), np.array([can.body, can.body, can.height]) / 2),
            (np.array([handle_x, 0.0, z0]), np.array([can.handle, can.handle, can.handle_length]) / 2),
            (np.array([handle_x + can.handle / 2 + can.gap / 2, 0.0, arm_z]), np.array([can.gap, can.handle, 0.010]) / 2)]


def arm_clear_of_can(kin, q: dict, T_hand_in_can: np.ndarray, can: CanGeometry, margin: float = 0.004,
                     links=("r_upper_arm", "r_elbow_1", "r_forearm")) -> bool:
    """False if the bending wrist would swing the arm into the can it holds. The hand itself is left out: it grips."""
    T_can = kin.hand(q) @ np.linalg.inv(T_hand_in_can)
    points = kin._points()
    boxes = can_boxes_in_can_frame(can)
    for name in links:
        pts, radius = points[name]
        T = kin.fk(name, q)
        w = (T[:3, :3] @ pts.T).T + T[:3, 3]
        local = (w - T_can[:3, 3]) @ T_can[:3, :3]
        for centre, half in boxes:
            d = np.maximum(np.abs(local - centre) - half, 0.0)
            if np.min(np.linalg.norm(d, axis=1)) < radius + margin:
                return False
    return True


def arm_clear_of_pots(kin, plan, pots_xy: list, pot_outer: float, rim_z: float, margin: float = 0.010) -> bool:
    """False if the arm would meet a pot anywhere on the way to the grasp."""
    return all(arm_clear_at(kin, q, pots_xy, pot_outer, rim_z, margin) for q in plan.path_reach + plan.path_approach)


def touches_scenery(point, pots: list, table_z: float, tolerance: float = 0.006) -> bool:
    if point.z < table_z + tolerance:
        return True
    for q in pots:
        r = math.hypot(point.x - q.x, point.y - q.y)
        if point.z < q.z + q.height + tolerance and q.radius - tolerance < r < q.radius + q.wall + tolerance:
            return True
    return False


def open_hand_clear(scene, can, can_yaw_deg: float, pot_xy, pot_outer: float, pre_distance: float = 0.07, margin: float = 0.010) -> bool:
    """False if the open hand would hit this pot on its way in. The hand is low enough to meet the pots
    from the pre-grasp to the grasp; above the pre-grasp it has risen clear of their rims."""
    R = rpy_matrix(0.0, 0.0, math.radians(can_yaw_deg + SEAT_OFFSET_DEG)) @ scene.R0
    a = scene.yaw0 + math.radians(can_yaw_deg)
    handle = scene.body_centre - np.array([math.cos(a), math.sin(a), 0.0]) * can.handle_to_body_centre
    grasp = handle - R @ scene.handle_in_hand
    pts = (R @ OPEN_HAND_POINTS.T).T
    for s in np.linspace(0.0, 1.0, 8):
        w = pts + grasp + R[:, 1] * (pre_distance * s)
        if np.min(np.hypot(w[:, 0] - pot_xy[0], w[:, 1] - pot_xy[1])) < pot_outer + OPEN_HAND_RADIUS + margin:
            return False
    return True


def finish(res: Result, state: dict, wall0: float, c: dict, contacts: dict) -> Result:
    res.contacts = [f"{k} ({v:.0f} N)" for k, v in contacts.items()]
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
    """The can tilted about the spout, as a person pours: the spout stays put and the body rises behind it."""
    c, s = math.cos(tilt), math.sin(tilt)
    Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    T = np.eye(4)
    T[:3, :3] = can_upright[:3, :3] @ Ry
    T[:3, 3] = can_upright[:3, :3] @ (SPOUT - Ry @ SPOUT) + can_upright[:3, 3]
    return T


def upright_over_pot(pot: Pot, heading: float) -> np.ndarray:
    """The upright can with its spout over the pot's centre."""
    T = np.eye(4)
    T[:3, :3] = rpy_matrix(0.0, 0.0, heading)
    T[:3, 3] = np.array([pot.x, pot.y, pot.z + pot.height + SPOUT_ABOVE_RIM]) - T[:3, :3] @ SPOUT
    return T


def pour_headings(kin, pot: Pot, pots_xy: list, T_hand_in_can: np.ndarray, q_start: dict, heading_now: float,
                  pot_outer: float, rim_z: float, can: CanGeometry, tilts_deg=(0, 30, 60, 75, 88), tolerance: float = 0.006) -> list:
    """Can headings for pouring over this pot, best first: (furthest tilt deg, turn rad, error m, heading).

    A heading counts only if the arm misses both pots and wrist_yaw keeps its margin all through the
    tilt. Better means it tilts further, then turns the can less on the way.
    """
    lo, hi = kin.limits(WRIST_YAW)
    rng = np.random.default_rng(0)
    out = {}
    for deg in range(-180, 180, 15):
        heading = math.radians(deg)
        upright = upright_over_pot(pot, heading)
        for k in range(3):
            q = dict(q_start)
            if k:
                for n in PLAN_JOINTS:
                    a, b = kin.limits(n)
                    q[n] = float(np.clip(q[n] + rng.normal(0, 0.1 if n.startswith("torso") else 0.3), a, b))
            q, e, r = kin.ik(q, upright @ T_hand_in_can, PLAN_JOINTS, CENTER_WEIGHTS, iterations=50, rest=REST)
            if (e + r * 0.1 > tolerance or not arm_clear_at(kin, q, pots_xy, pot_outer, rim_z)
                    or not arm_clear_of_can(kin, q, T_hand_in_can, can)):
                continue
            margin = min(q[WRIST_YAW] - lo, hi - q[WRIST_YAW])
            furthest, worst = 0, e
            for t in tilts_deg[1:]:
                q, e, r = kin.ik(q, pour_pose(upright, math.radians(t)) @ T_hand_in_can, PLAN_JOINTS, CENTER_WEIGHTS, iterations=25, rest=REST)
                if (e + r * 0.1 > tolerance or not arm_clear_at(kin, q, pots_xy, pot_outer, rim_z)
                        or not arm_clear_of_can(kin, q, T_hand_in_can, can)):
                    break
                margin = min(margin, q[WRIST_YAW] - lo, hi - q[WRIST_YAW])
                furthest, worst = t, max(worst, e)
            if margin < math.radians(MIN_POUR_MARGIN_DEG):
                continue
            turn = abs((heading - heading_now + math.pi) % (2 * math.pi) - math.pi)
            if deg not in out or (furthest, -turn, -worst) > (out[deg][0], -out[deg][1], -out[deg][2]):
                out[deg] = (furthest, turn, worst, heading)
    return sorted(out.values(), key=lambda c: (-c[0], c[1], c[2]))
