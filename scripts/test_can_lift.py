#!/usr/bin/env python3
"""Lift test: a can-shaped object with a side handle stands on a table, gravity stays on.

iCub's arm settles first, then the table and can appear with the handle inside the open hand (this
skips the reach and approach, which cuRobo will do later). The hand closes, then the arm lifts straight
up. The can is held by contact and friction only.

The can is one floating articulation of 4 boxes: body, handle bar and two connecting arms, joined by
locked hinges. threepp's add() gives only one shape per body, and this version has no fixed joint.

Reports how far the can is dragged while the hand closes, and whether it follows the lift.

Usage:
  python scripts/test_can_lift.py --finger-limit 1
  python scripts/test_can_lift.py --finger-limit 1 --closure 0     # control: the can must stay down

Nothing is written to disk.
"""
import argparse
import math
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import threepp as tp
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit_hand_colliders import BASE_POSITION, quat_to_matrix, rpy_to_matrix, urdf_link_frames  # noqa: E402
from gardener.sim.robot import FINGERS, ICub, RIGHT_ARM, TORSO, make_world  # noqa: E402
from test_grasp_contacts import PARTS, POSES, link_pose  # noqa: E402

DEFAULT_URDF = "~/icub-assets/iCub/robots/iCubGazeboV2_10/gardener_hand.urdf"
DT = 0.02             # seconds of simulation per control step
SETTLE_S = 2.0        # arm settles before the can appears
CAN_SETTLE_S = 0.5    # can rests on the table before the hand closes
HOLD_S = 1.0          # hold at full closure before lifting
LIFT_S = 2.0          # duration of the lift
AFTER_LIFT_S = 3.0    # hold in the air
SUCCESS_MIN_RISE = 0.020   # the can must rise at least this much (m)
SUCCESS_MAX_SLIP = 0.005   # and move at most this much relative to the hand during the lift (m)
SUCCESS_MAX_TILT = 15.0    # and tilt at most this much in the air (deg)


class ArmKinematics:
    """Forward kinematics of the URDF chain from the base to r_hand, and a small IK solver."""

    def __init__(self, urdf, base):
        root = ET.parse(urdf).getroot()
        joints = {j.find("child").get("link"): j for j in root.findall("joint")}
        chain, name = [], "r_hand"
        while name in joints:
            chain.append(joints[name])
            name = joints[name].find("parent").get("link")
        self.chain = chain[::-1]
        self.base = np.array(base, dtype=float)

    def hand(self, q):
        T = np.eye(4)
        T[:3, 3] = self.base
        for j in self.chain:
            o = j.find("origin")
            L = np.eye(4)
            if o is not None:
                L[:3, :3] = rpy_to_matrix(*[float(v) for v in o.get("rpy", "0 0 0").split()])
                L[:3, 3] = [float(v) for v in o.get("xyz", "0 0 0").split()]
            T = T @ L
            if j.get("type") != "fixed" and j.get("name") in q:
                a = np.array([float(v) for v in j.find("axis").get("xyz").split()])
                a /= np.linalg.norm(a)
                K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
                M = np.eye(4)
                t = q[j.get("name")]
                M[:3, :3] = np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * K @ K
                T = T @ M
        return T

    def solve_lift(self, q0, names, lower, upper, dz):
        """Joint angles that raise the hand by dz with the same orientation (damped least squares)."""
        q = dict(q0)
        T0 = self.hand(q)
        goal_p, goal_R = T0[:3, 3] + np.array([0, 0, dz]), T0[:3, :3]

        def error(qd):
            T = self.hand(qd)
            dR = goal_R @ T[:3, :3].T
            rot = 0.5 * np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])
            return np.concatenate([goal_p - T[:3, 3], rot])

        for _ in range(30):
            e = error(q)
            if np.linalg.norm(e) < 1e-5:
                break
            J = np.zeros((6, len(names)))
            for i, n in enumerate(names):
                d = dict(q)
                d[n] += 1e-5
                J[:, i] = (e - error(d)) / 1e-5
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
            for i, n in enumerate(names):
                q[n] = float(np.clip(q[n] + dq[i], lower[n], upper[n]))
        return q, float(np.linalg.norm(error(q)))


def box_mesh(size, pos, yaw):
    mesh = tp.Mesh(tp.BoxGeometry(*map(float, size)), tp.MeshStandardMaterial())
    mesh.position.set(*map(float, pos))
    mesh.quaternion.set(0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
    return mesh


def build_can(world, handle_pos, u, table_z, a, material):
    """Body on the far side of the handle along horizontal direction u. Returns (articulation, body link, meshes)."""
    yaw = math.atan2(u[1], u[0])
    h, hw = a.handle / 1000.0, a.handle_length / 1000.0
    gap, bd, bw, bh = a.gap / 1000.0, a.body / 1000.0, a.body / 1000.0, a.height / 1000.0
    arm_t = 0.010  # thickness of the two connecting arms
    # Hollow-can realism is not needed here: mass goes to the body, little to the handle parts.
    handle_mass = 0.010
    arm_len = gap + 0.002
    body_c = handle_pos + u * (h / 2 + gap + bd / 2)
    body_c[2] = table_z + bh / 2
    body_mass = a.mass - handle_mass - 2 * 0.005
    art = world.create_articulation(fixed_base=False, solver_position_iterations=16,
                                    disable_self_collision=True, drive_limits_are_forces=True)
    body_mesh = box_mesh((bd, bw, bh), body_c, yaw)
    body = art.add_link(body_mesh, density=body_mass / (bd * bw * bh), material=material)
    meshes = {"body": body_mesh}

    def locked(name, size, pos, mass):
        mesh = box_mesh(size, pos, yaw)
        art.add_link(mesh, parent=body, density=mass / (size[0] * size[1] * size[2]), axis=[0.0, 0.0, 1.0],
                     anchor=[float(v) for v in pos], lower=-1e-5, upper=1e-5, material=material)
        meshes[name] = mesh

    locked("handle", (h, h, hw), handle_pos, handle_mass)
    for name, sign in (("arm_top", 1), ("arm_bottom", -1)):
        p = handle_pos + u * (h / 2 + gap / 2) + np.array([0.0, 0.0, sign * (hw / 2 - arm_t / 2)])
        locked(name, (arm_len, h, arm_t), p, 0.005)
    art.finalize()
    return art, body, meshes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--urdf", default=DEFAULT_URDF)
    ap.add_argument("--finger-limit", type=float, required=True, help="torque limit for the hand joints (N*m)")
    ap.add_argument("--finger-stiffness", type=float, default=20.0, help="N*m/rad")
    ap.add_argument("--finger-damping", type=float, default=0.1, help="N*m*s/rad")
    ap.add_argument("--pos", type=float, nargs=3, default=[-66.0, -22.0, 35.0], metavar=("X", "Y", "Z"),
                    help="handle centre in the r_hand frame (mm); closer than x = -63 the thumb and forearm hit things")
    ap.add_argument("--handle", type=float, default=20.0, help="handle width and depth (mm)")
    ap.add_argument("--handle-length", type=float, default=110.0,
                    help="handle length (mm); long enough that the connecting arms clear the fingers")
    ap.add_argument("--gap", type=float, default=50.0, help="gap between handle and body, for the fingers (mm)")
    ap.add_argument("--body", type=float, default=70.0, help="can body width and depth (mm)")
    ap.add_argument("--height", type=float, default=125.0, help="can body height (mm)")
    ap.add_argument("--mass", type=float, default=0.20, help="total can mass (kg)")
    ap.add_argument("--friction", type=float, default=0.5, help="upper limit on friction at every can contact")
    ap.add_argument("--closure", type=float, default=1.0, help="final hand closure; 0 is the control")
    ap.add_argument("--close-time", type=float, default=2.0, help="seconds to close the hand")
    ap.add_argument("--no-can", action="store_true", help="same arm motion with no table or can: measures the arm alone")
    ap.add_argument("--lift", type=float, default=30.0,
                    help="commanded lift (mm); under a firm grip the hand falls about 3-8 mm short")
    ap.add_argument("--timestep", type=float, default=0.0005,
                    help="physics substep (s); at 2 ms a firm grip disturbs the arm by about 2 deg, at 0.5 ms it does not")
    ap.add_argument("--solver-iters", type=int, default=12, help="robot solver position iterations, for convergence checks")
    args = ap.parse_args()

    urdf = os.path.expanduser(args.urdf)
    world = make_world(timestep=args.timestep, max_substeps=math.ceil(DT / args.timestep) + 1)
    robot = ICub(world, urdf, base_position=BASE_POSITION, solver_iterations=args.solver_iters,
                 finger_drive=(args.finger_stiffness, args.finger_damping, args.finger_limit))
    world.step(1e-4)
    root = ET.parse(urdf).getroot()
    hand_link = robot.link("r_hand")
    T_com_to_urdf = np.linalg.inv(link_pose(hand_link)) @ urdf_link_frames(root, BASE_POSITION)["r_hand"]

    def hand_T():
        return link_pose(hand_link) @ T_com_to_urdf

    sensors = {}
    for name in PARTS + ["r_forearm", "r_wrist_1", "r_elbow_1"]:
        sensors[name] = tp.ContactSensor(robot.collider(name))
        world.register_sensor(sensors[name])

    with open(POSES) as f:
        pose = yaml.safe_load(f)["poses"]["reach_forward"]
    robot.set_targets({j: math.radians(v) for j, v in pose.items()})
    robot.set_targets(robot.hand_targets(0.0))
    robot.apply()
    for _ in range(int(SETTLE_S / DT)):
        world.step(DT)
    print(f"arm settled: worst joint err {math.degrees(robot.tracking_error()):.2f} deg")

    # Plan the lift now, using the arm's actual angles.
    kin = ArmKinematics(urdf, BASE_POSITION)
    chain_names = TORSO + RIGHT_ARM
    q_now = dict(zip(chain_names, robot.get(chain_names)))
    T_fk, T_sim = kin.hand(q_now), hand_T()
    print(f"kinematics check: FK and simulated hand differ by {np.linalg.norm(T_fk[:3, 3] - T_sim[:3, 3]) * 1000:.2f} mm")
    lower = {n: robot.lower[robot.index[n]] for n in RIGHT_ARM}
    upper = {n: robot.upper[robot.index[n]] for n in RIGHT_ARM}
    q_up, ik_err = kin.solve_lift(q_now, RIGHT_ARM, lower, upper, args.lift / 1000.0)
    arm_start = {n: robot.targets[robot.index[n]] for n in RIGHT_ARM}
    arm_goal = {n: arm_start[n] + q_up[n] - q_now[n] for n in RIGHT_ARM}
    print(f"lift plan: IK error {ik_err * 1000:.2f} mm, joint moves (deg): "
          + ", ".join(f"{n[2:]} {math.degrees(arm_goal[n] - arm_start[n]):+.1f}" for n in RIGHT_ARM))

    def report_arm(label):
        """Print each arm joint's error and the hand's position, so drift can be followed over time."""
        p = robot.positions()
        errs = ", ".join(f"{n[2:]} {math.degrees(robot.targets[robot.index[n]] - p[robot.index[n]]):+.2f}" for n in RIGHT_ARM)
        print(f"  {label:<18} hand z {hand_T()[2, 3] * 1000:8.2f} mm | err (deg): {errs}")

    if args.no_can:
        print("no table, no can. Arm joint error = target - actual:")
        report_arm("settled")
        start_z, closure = hand_T()[2, 3], 0.0
        for step in range(int(args.close_time / DT)):
            closure = args.closure * (step + 1) * DT / args.close_time
            robot.set_targets(robot.hand_targets(closure))
            robot.apply()
            world.step(DT)
        for _ in range(int(HOLD_S / DT)):
            world.step(DT)
        report_arm("hand closed" if args.closure > 0 else "before lift")
        closed_z = hand_T()[2, 3]
        for step in range(int(LIFT_S / DT)):
            s = (step + 1) * DT / LIFT_S
            s = s * s * (3 - 2 * s)
            robot.set_targets({n: arm_start[n] + s * (arm_goal[n] - arm_start[n]) for n in RIGHT_ARM})
            robot.apply()
            world.step(DT)
        report_arm("end of lift")
        for _ in range(int(AFTER_LIFT_S / DT)):
            world.step(DT)
        report_arm(f"{AFTER_LIFT_S:.0f} s later")
        print(f"HAND RISE: {(hand_T()[2, 3] - closed_z) * 1000:.2f} mm of {args.lift:.0f} mm commanded "
              f"(hand moved {(closed_z - start_z) * 1000:+.2f} mm while closing)")
        return

    # Place the table and can around the settled hand.
    T = hand_T()
    handle_pos = (T @ np.append(np.array(args.pos) / 1000.0, 1.0))[:3]
    u = -T[:3, 1].copy()         # fingers close towards -y of the hand: the body goes on that side
    u[2] = 0.0
    u /= np.linalg.norm(u)
    v = np.array([-u[1], u[0], 0.0])
    # The handle sits above the table so that its centre is at the requested height.
    bottom_clearance = (args.height - args.handle_length) / 2000.0
    table_z = handle_pos[2] - args.handle_length / 2000.0 - bottom_clearance
    h, gap, bd = args.handle / 1000.0, args.gap / 1000.0, args.body / 1000.0
    # A table under the can's body only: the forearm collider is a big box and must not hit it.
    near, far = h / 2 + gap - 0.005, h / 2 + gap + bd + 0.005
    table_c = handle_pos + u * (near + far) / 2
    table_c[2] = table_z - 0.01
    table_mesh = box_mesh((far - near, bd, 0.02), table_c, math.atan2(u[1], u[0]))
    material = world.create_material(args.friction, args.friction, 0.0, friction_combine="min")
    world.add_static(table_mesh, material=material)
    art_can, body, can_meshes = build_can(world, handle_pos, u, table_z, args, material)
    can_sensors = {n: tp.ContactSensor(m) for n, m in can_meshes.items()}
    for s in can_sensors.values():
        world.register_sensor(s)

    first_touch, peak, closure = {}, {n: 0.0 for n in sensors}, 0.0
    hold_sum, hold_n = {n: 0.0 for n in sensors}, 0
    can_peak = {n: 0.0 for n in can_sensors}
    can_hold = {n: 0.0 for n in can_sensors}
    can_first = {}

    def tick(in_hold=False):
        nonlocal hold_n
        robot.set_targets(robot.hand_targets(closure))
        robot.apply()
        world.step(DT)
        for name, sensor in sensors.items():
            force = 0.0
            for s in sensor.drain():
                f = s.force
                force = max(force, math.sqrt(f.x * f.x + f.y * f.y + f.z * f.z))
            if force > 0.0 and name not in first_touch:
                first_touch[name] = closure
            peak[name] = max(peak[name], force)
            if in_hold:
                hold_sum[name] += force
        for name, sensor in can_sensors.items():
            force = 0.0
            for s in sensor.drain():
                f = s.force
                force = max(force, math.sqrt(f.x * f.x + f.y * f.y + f.z * f.z))
            if name != "body" and force > 0.0 and name not in can_first:
                can_first[name] = closure
            can_peak[name] = max(can_peak[name], force)
            if in_hold:
                can_hold[name] += force
        if in_hold:
            hold_n += 1

    def can_state():
        p, q = body.position, body.quaternion
        R = quat_to_matrix(q.w, q.x, q.y, q.z)
        tilt = math.degrees(math.acos(float(np.clip(R[2, 2], -1.0, 1.0))))
        yaw = math.degrees(math.atan2(R[1, 0], R[0, 0]))
        return np.array([p.x, p.y, p.z]), tilt, yaw

    # Let the can rest on the table with the hand open.
    for _ in range(int(CAN_SETTLE_S / DT)):
        tick()
    touching_open = sorted(first_touch)
    first_touch.clear()
    rest_p, rest_tilt, rest_yaw = can_state()
    print(f"can resting: tilt {rest_tilt:.2f} deg, hand parts touching with the hand open: {touching_open or 'nothing'}")
    report_arm("before closing")

    # Close.
    for step in range(int(args.close_time / DT)):
        closure = args.closure * (step + 1) * DT / args.close_time
        tick()
    closure = args.closure
    for _ in range(int(HOLD_S / DT)):
        tick(in_hold=True)
    closed_p, closed_tilt, closed_yaw = can_state()
    report_arm("hand closed")
    drag_xy = float(np.linalg.norm((closed_p - rest_p)[:2])) * 1000
    print(f"\nDRAG while closing: {drag_xy:.1f} mm on the table, height change {(closed_p - rest_p)[2] * 1000:+.1f} mm, "
          f"yaw change {closed_yaw - rest_yaw:+.1f} deg, tilt {closed_tilt:.1f} deg")

    # Lift, planned from where the closed hand really is: pressing the can on the table pushes the arm
    # up a few mm, and a plan from the settled pose would lose that from the lift. Each joint keeps the
    # settled gap between target and actual angle, which is what holds the arm up against gravity.
    q_closed = dict(zip(chain_names, robot.get(chain_names)))
    q_up, ik_err = kin.solve_lift(q_closed, RIGHT_ARM, lower, upper, args.lift / 1000.0)
    arm_goal = {n: q_up[n] + arm_start[n] - q_now[n] for n in RIGHT_ARM}
    print(f"lift replanned from the closed hand: IK error {ik_err * 1000:.2f} mm")
    hand_before, can_before = hand_T()[:3, 3].copy(), closed_p.copy()
    rel_before = np.linalg.inv(hand_T()) @ np.append(closed_p, 1.0)
    handle_mesh = can_meshes["handle"]

    def handle_in_hand():
        p = handle_mesh.position
        return (np.linalg.inv(hand_T()) @ np.array([p.x, p.y, p.z, 1.0]))[:3] * 1000

    handle_before = handle_in_hand()
    for step in range(int(LIFT_S / DT)):
        s = (step + 1) * DT / LIFT_S
        s = s * s * (3 - 2 * s)
        robot.set_targets({n: arm_start[n] + s * (arm_goal[n] - arm_start[n]) for n in RIGHT_ARM})
        tick()
    report_arm("end of lift")
    for _ in range(int(AFTER_LIFT_S / DT)):
        tick()
    report_arm(f"{AFTER_LIFT_S:.0f} s later")
    end_p, end_tilt, _ = can_state()
    # Height of the body's lowest corner above the table: positive means it has really left the table.
    q_end = body.quaternion
    half = np.array([args.body, args.body, args.height]) / 2000.0
    corner_drop = float(np.abs(quat_to_matrix(q_end.w, q_end.x, q_end.y, q_end.z)[2, :]) @ half)
    clearance = (float(end_p[2]) - corner_drop - table_z) * 1000
    hand_rise = float(hand_T()[2, 3] - hand_before[2]) * 1000
    can_rise = float(end_p[2] - can_before[2]) * 1000
    rel_after = np.linalg.inv(hand_T()) @ np.append(end_p, 1.0)
    slip = float(np.linalg.norm((rel_after - rel_before)[:3])) * 1000
    ok = (can_rise >= SUCCESS_MIN_RISE * 1000 and slip <= SUCCESS_MAX_SLIP * 1000 and end_tilt <= SUCCESS_MAX_TILT)

    handle_slip = float(np.linalg.norm(handle_in_hand() - handle_before))
    print(f"\nLIFT: hand rose {hand_rise:.1f} mm, can rose {can_rise:.1f} mm, can body moved {slip:.1f} mm relative to the hand "
          f"(handle itself {handle_slip:.1f} mm), tilt {end_tilt:.1f} deg")
    arm_err = {n[2:]: math.degrees(robot.targets[robot.index[n]] - robot.positions()[robot.index[n]]) for n in RIGHT_ARM}
    print("arm joint error at the end (target - actual, deg): " + ", ".join(f"{n} {e:+.2f}" for n, e in arm_err.items()))
    print(f"\n{'part':<20}{'first touch at closure':>24}{'peak force (N)':>16}{'hold force (N)':>16}")
    for name in sensors:
        if name in first_touch or peak[name] > 0.0:
            print(f"{name:<20}{first_touch.get(name, float('nan')):>24.2f}{peak[name]:>16.1f}{hold_sum[name] / max(hold_n, 1):>16.1f}")
    print(f"\n{'can part':<20}{'first touch at closure':>24}{'peak force (N)':>16}{'hold force (N)':>16}")
    for name in can_sensors:
        print(f"{name:<20}{can_first.get(name, float('nan')):>24.2f}{can_peak[name]:>16.1f}{can_hold[name] / max(hold_n, 1):>16.1f}")
    print("(the body's force includes the table; its first touch is not tracked)")
    print(f"lowest corner of the can body above the table at the end: {clearance:.1f} mm")
    missing = [f for f in FINGERS if not any(n in first_touch for n in PARTS if f"_{f}_" in n)]
    print(f"\nfingers with no contact at all: {missing or 'none'}")
    print(f"can: {args.mass} kg, handle {args.handle:.0f} mm, friction limit {args.friction}, finger limit {args.finger_limit} N*m")
    print(f"RESULT: {'LIFT SUCCESS' if ok else 'LIFT FAILED'} (needs rise >= {SUCCESS_MIN_RISE * 1000:.0f} mm, "
          f"slip <= {SUCCESS_MAX_SLIP * 1000:.0f} mm, tilt <= {SUCCESS_MAX_TILT:.0f} deg)")


if __name__ == "__main__":
    main()
