#!/usr/bin/env python3
"""Reach, approach and lift: the can stands on the table and the hand starts away from it.

Sequence: hand above and behind the pre-grasp, reach to the pre-grasp, straight approach with the thumb
opposed but uncurled until the palm is close to the handle, wrap the fingers, curl the thumb last, then lift.
The can is held by contact and friction only.

Reports drag at each stage, lift success and the wrist_yaw margin for one can rotation.

Usage:
  python scripts/test_reach_grasp.py --can-yaw 25
  python scripts/test_reach_grasp.py --no-grasp      # control: the hand stays open and the can stays down

Nothing is written to disk.
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import threepp as tp
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gardener.expert.planner import PLAN_JOINTS, WRIST_YAW, nominal_scene, plan_grasp  # noqa: E402
from gardener.sim.kinematics import ArmKinematics, quat_matrix  # noqa: E402
from gardener.sim.robot import DEFAULT_FINGER_DRIVE, FINGERS, ICub, RIGHT_ARM, make_world  # noqa: E402
from gardener.sim.scene import TABLE_BACK_X, CanGeometry, add_table, build_can  # noqa: E402

DEFAULT_URDF = "~/icub-assets/iCub/robots/iCubGazeboV2_10/gardener_hand.urdf"
POSES = os.path.join(ROOT, "configs", "poses.yaml")
BASE_POSITION = (0.0, 0.0, 0.05)
DT = 0.02
HAND_PARTS = ["r_hand"] + [f"r_hand_{f}_{k}" for f in FINGERS for k in range(4)]
SUCCESS_MIN_RISE, SUCCESS_MAX_SLIP, SUCCESS_MAX_TILT = 20.0, 5.0, 15.0   # mm, mm, deg (same as test_can_lift.py)
TARGET_DRAG = 10.0  # mm


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--urdf", default=DEFAULT_URDF)
    ap.add_argument("--can-yaw", type=float, default=0.0, help="can rotation about its body centre (deg)")
    ap.add_argument("--handle-in-hand", type=float, nargs=3, default=[-70.0, -21.0, 12.5], metavar=("X", "Y", "Z"),
                    help="handle centre in the r_hand frame at the grasp (mm)")
    ap.add_argument("--finger-limit", type=float, default=DEFAULT_FINGER_DRIVE[2], help="N*m")
    ap.add_argument("--mass", type=float, default=0.20, help="can mass (kg)")
    ap.add_argument("--friction", type=float, default=0.5, help="upper limit on friction at every can contact")
    ap.add_argument("--no-grasp", action="store_true", help="control: never close the hand")
    ap.add_argument("--lift", type=float, default=30.0, help="commanded lift (mm)")
    ap.add_argument("--finger-time", type=float, default=2.0, help="seconds to wrap the fingers")
    ap.add_argument("--thumb-time", type=float, default=1.5, help="seconds for a full thumb curl")
    ap.add_argument("--thumb-curl", type=float, default=0.7, help="final thumb curl, 0 to 1")
    ap.add_argument("--yaw-shift", type=float, default=15.0,
                    help="turns the nominal can and hand (deg); +15 centres the arm's range on the can's +-25 deg")
    ap.add_argument("--handle-length", type=float, default=85.0, help="mm; ends below the open thumb tip")
    ap.add_argument("--height", type=float, default=100.0, help="can body height (mm)")
    ap.add_argument("--top-arm", action="store_true", help="also join the handle to the body at the top")
    ap.add_argument("--seat-offset", type=float, default=-3.0, help="hand yaw relative to the can (deg)")
    ap.add_argument("--opposition", type=float, default=1.0, help="how far the thumb swings out, 0 to 1")
    args = ap.parse_args()

    urdf = os.path.expanduser(args.urdf)
    can = CanGeometry(handle_length=args.handle_length / 1000.0, height=args.height / 1000.0, mass=args.mass,
                      top_arm=args.top_arm)
    with open(POSES) as f:
        pose = yaml.safe_load(f)["poses"]["grasp_table"]
    q_nominal = {n: math.radians(pose.get(n, 0.0)) for n in PLAN_JOINTS}
    kin = ArmKinematics(urdf, base=BASE_POSITION)
    scene = nominal_scene(kin, q_nominal, np.array(args.handle_in_hand) / 1000.0, can, args.yaw_shift)
    t0 = time.perf_counter()
    plan = plan_grasp(kin, scene, can, math.radians(args.can_yaw), q_nominal, TABLE_BACK_X, lift=args.lift / 1000.0,
                      seat_offset_deg=args.seat_offset)
    if plan is None:
        print(f"RESULT: NO PLAN for can yaw {args.can_yaw} deg")
        return
    lo, hi = kin.limits(WRIST_YAW)
    print(f"plan ({time.perf_counter() - t0:.1f} s): hand yaw offset {plan.hand_yaw_offset_deg:+.0f} deg, path error "
          f"{plan.position_error * 1000:.2f} mm, wrist_yaw margin {math.degrees(plan.wrist_yaw_margin):.1f} deg, "
          f"forearm clearance {plan.min_clearance['r_forearm'] * 1000:.1f} mm")

    world = make_world()
    robot = ICub(world, urdf, base_position=BASE_POSITION,
                 finger_drive=(DEFAULT_FINGER_DRIVE[0], DEFAULT_FINGER_DRIVE[1], args.finger_limit))
    world.step(1e-4)

    sensors = {}
    for name in robot.body_names:
        sensors[name] = tp.ContactSensor(robot.collider(name))
        world.register_sensor(sensors[name])

    fingers, thumb = 0.0, 0.0
    worst_margin = math.inf
    first_touch, peak, physics_wall, sim_time = {}, {}, 0.0, 0.0
    phase = "reach"

    def step():
        nonlocal worst_margin, physics_wall, sim_time
        robot.set_targets(robot.grasp_targets(fingers, thumb, args.opposition))
        robot.apply()
        t = time.perf_counter()
        world.step(DT)
        physics_wall += time.perf_counter() - t
        sim_time += DT
        for name, sensor in sensors.items():
            force = 0.0
            for s in sensor.drain():
                f = s.force
                force = max(force, math.sqrt(f.x * f.x + f.y * f.y + f.z * f.z))
            if force > 0.0:
                first_touch.setdefault(name, phase)
                peak[name] = max(peak.get(name, 0.0), force)
        y = robot.get([WRIST_YAW])[0]
        worst_margin = min(worst_margin, y - lo, hi - y)

    def hold(seconds):
        for _ in range(int(seconds / DT)):
            step()

    # Settle at the ready pose, away from where the can will be. The gap between target and actual
    # angle at rest is what holds the arm up against gravity, so every later target keeps it.
    robot.set_targets({n: plan.q_ready[n] for n in PLAN_JOINTS})
    hold(3.0)
    measured = dict(zip(PLAN_JOINTS, robot.get(PLAN_JOINTS)))
    sag = {n: robot.targets[robot.index[n]] - measured[n] for n in PLAN_JOINTS}

    material = world.create_material(args.friction, args.friction, 0.0, friction_combine="min")
    add_table(world, scene.table_z, material)
    a = scene.yaw0 + math.radians(args.can_yaw)
    u = np.array([math.cos(a), math.sin(a), 0.0])
    handle_pos = scene.body_centre - u * can.handle_to_body_centre
    body, can_meshes = build_can(world, handle_pos, u, scene.table_z, can, material)
    can_sensors = {n: tp.ContactSensor(m) for n, m in can_meshes.items()}
    for s in can_sensors.values():
        world.register_sensor(s)

    def can_yaw_now():
        q = body.quaternion
        R = quat_matrix(q.w, q.x, q.y, q.z)
        return math.degrees(math.atan2(R[1, 0], R[0, 0]))

    def can_state():
        p, q = body.position, body.quaternion
        R = quat_matrix(q.w, q.x, q.y, q.z)
        return np.array([p.x, p.y, p.z]), math.degrees(math.acos(float(np.clip(R[2, 2], -1, 1))))

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

    drag_log = []

    def log_drag(label):
        p, tilt = can_state()
        turn = (can_yaw_now() - rest_yaw + 180.0) % 360.0 - 180.0
        drag_log.append((label, float(np.linalg.norm((p - rest_p)[:2])) * 1000, tilt, turn))

    hold(0.5)
    rest_p, rest_tilt = can_state()
    rest_yaw = can_yaw_now()
    print(f"can resting: tilt {rest_tilt:.2f} deg; touching anything on the robot: {sorted(first_touch) or 'nothing'}")

    phase = "reach"
    follow(plan.path_reach, 2.0)
    log_drag("after reach")
    phase = "approach"
    follow(plan.path_approach, 2.0)
    hold(0.3)
    log_drag("after approach")

    if not args.no_grasp:
        phase = "fingers"
        n = int(args.finger_time / DT)
        for i in range(n):
            fingers = (i + 1) / n
            step()
        hold(0.3)
        log_drag("after fingers")
        phase = "thumb"
        n = max(1, int(args.thumb_time * args.thumb_curl / DT))
        for i in range(n):
            thumb = args.thumb_curl * (i + 1) / n
            step()
        hold(1.0)
        log_drag("after thumb")
    closed_p, closed_tilt = can_state()
    print("drag while grasping (mm on the table, tilt deg, can turn deg): "
          + "; ".join(f"{label} {d:.1f}, {t:.1f}, {turn:+.1f}" for label, d, t, turn in drag_log))

    # Lift, planned from where the hand really is.
    phase = "lift"
    q_closed = dict(zip(PLAN_JOINTS, robot.get(PLAN_JOINTS)))
    T = kin.hand(q_closed)
    hand_z0 = T[2, 3]
    rel0 = np.linalg.inv(T) @ np.append(closed_p, 1.0)
    handle_mesh = can_meshes["handle"]

    def handle_in_hand(Th):
        p = handle_mesh.position
        return (np.linalg.inv(Th) @ np.array([p.x, p.y, p.z, 1.0]))[:3] * 1000

    h0 = handle_in_hand(T)
    print(f"handle in the hand after closing: {np.round(h0, 1)} mm (started at {args.handle_in_hand})")
    path, q = [], dict(q_closed)
    for s in np.linspace(0.0, 1.0, 13)[1:]:
        Tg = T.copy()
        Tg[2, 3] = hand_z0 + s * args.lift / 1000.0
        q, _, _ = kin.ik(q, Tg, PLAN_JOINTS, iterations=25)
        path.append(dict(q))
    follow(path, 2.0)
    hold(3.0)

    q_end = dict(zip(PLAN_JOINTS, robot.get(PLAN_JOINTS)))
    Tend = kin.hand(q_end)
    end_p, end_tilt = can_state()
    hand_rise = (Tend[2, 3] - hand_z0) * 1000
    can_rise = (end_p[2] - closed_p[2]) * 1000
    body_slip = float(np.linalg.norm((np.linalg.inv(Tend) @ np.append(end_p, 1.0) - rel0)[:3])) * 1000
    handle_slip = float(np.linalg.norm(handle_in_hand(Tend) - h0))
    drag = drag_log[-1][1]

    turn_in_air = (can_yaw_now() - rest_yaw + 180.0) % 360.0 - 180.0 - drag_log[-1][3]
    print(f"can turn during the lift: {turn_in_air:+.1f} deg")
    print(f"LIFT: hand rose {hand_rise:.1f} mm, can rose {can_rise:.1f} mm, can body moved {body_slip:.1f} mm relative to the "
          f"hand (handle {handle_slip:.1f} mm), tilt {end_tilt:.1f} deg")
    print(f"WRIST: wrist_yaw margin planned {math.degrees(plan.wrist_yaw_margin):.1f} deg, measured minimum {math.degrees(worst_margin):.1f} deg")
    print(f"\n{'hand part':<20}{'first touch during':>20}{'peak force (N)':>16}")
    for name in HAND_PARTS:
        if name in first_touch:
            print(f"{name:<20}{first_touch[name]:>20}{peak[name]:>16.1f}")
    others = [n for n in first_touch if n not in HAND_PARTS]
    print("robot bodies outside the hand touching anything: "
          + (", ".join(f"{n} during {first_touch[n]} ({peak[n]:.0f} N)" for n in others) if others else "none"))
    touched = [n for n in HAND_PARTS if first_touch.get(n) in ("reach", "approach")]
    print(f"hand parts touching before the grasp starts: {touched or 'none'}")
    print(f"SPEED: {sim_time / physics_wall:.2f} x real time")
    ok = can_rise >= SUCCESS_MIN_RISE and body_slip <= SUCCESS_MAX_SLIP and end_tilt <= SUCCESS_MAX_TILT and not others
    print(f"RESULT: LIFT {'SUCCESS' if ok else 'FAILED'}, drag {drag:.1f} mm ({'under' if drag < TARGET_DRAG else 'OVER'} the "
          f"{TARGET_DRAG:.0f} mm target), can yaw {args.can_yaw} deg")


if __name__ == "__main__":
    main()
