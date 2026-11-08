#!/usr/bin/env python3
"""Close iCub's right hand on a simple bar and report which hand parts touch it, and how hard.

The bar's long axis lies along the hand's z axis (across the fingers). Its position is given in the
r_hand link frame, in mm. Fingers close towards -y of that frame.

Two modes:
  static   the bar cannot move, so this shows the real contact map (who touches, when, how hard).
  dynamic  a free bar of --mass kg. Gravity stays off while the hand closes, then turns on, and the
           report says how far the bar slipped inside the hand.

Usage:
  python scripts/test_grasp_contacts.py --mode static  --pos -85 -28 20
  python scripts/test_grasp_contacts.py --mode dynamic --pos -85 -28 20 --mass 0.15

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
from fit_hand_colliders import BASE_POSITION, matrix_to_rpy, quat_to_matrix, urdf_link_frames  # noqa: E402
from gardener.sim.robot import FINGERS, ICub, make_world  # noqa: E402

DEFAULT_URDF = "~/icub-assets/iCub/robots/iCubGazeboV2_10/gardener_hand.urdf"
POSES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "poses.yaml")
DT = 0.02             # seconds of simulation per control step
SETTLE_S = 2.0        # time for the arm to reach its pose before the bar appears
HOLD_S = 1.0          # time holding at full closure
GRAVITY_ON_S = 3.0    # dynamic mode: time with gravity on
PARTS = ["r_hand"] + [f"r_hand_{f}_{k}" for f in FINGERS for k in range(4)]


def link_pose(link):
    """4x4 world pose of a PhysX link (its centre-of-mass frame, not the URDF link frame)."""
    q, p = link.quaternion, link.position
    T = np.eye(4)
    T[:3, :3] = quat_to_matrix(q.w, q.x, q.y, q.z)
    T[:3, 3] = (p.x, p.y, p.z)
    return T


def make_bar(T_world, size_m, length_m):
    """A box mesh placed at T_world, with its long side along the local z axis."""
    mesh = tp.Mesh(tp.BoxGeometry(size_m, size_m, length_m), tp.MeshStandardMaterial())
    mesh.position.set(*map(float, T_world[:3, 3]))
    roll, pitch, yaw = matrix_to_rpy(T_world[:3, :3])
    # Same convention as URDF rpy: R = Rz(yaw) Ry(pitch) Rx(roll).
    cr, sr, cp, sp, cy, sy = (math.cos(roll / 2), math.sin(roll / 2), math.cos(pitch / 2),
                              math.sin(pitch / 2), math.cos(yaw / 2), math.sin(yaw / 2))
    mesh.quaternion.set(sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
                        cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy)
    q = mesh.quaternion
    if np.abs(quat_to_matrix(q.w, q.x, q.y, q.z) - T_world[:3, :3]).max() > 1e-4:
        raise RuntimeError("bar rotation does not match the requested pose (quaternion order?)")
    return mesh


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--urdf", default=DEFAULT_URDF)
    ap.add_argument("--mode", choices=["static", "dynamic"], default="static")
    ap.add_argument("--pos", type=float, nargs=3, default=[-85.0, -28.0, 20.0], metavar=("X", "Y", "Z"),
                    help="bar centre in the r_hand frame (mm)")
    ap.add_argument("--size", type=float, default=20.0, help="bar width and depth (mm)")
    ap.add_argument("--length", type=float, default=90.0, help="bar length (mm)")
    ap.add_argument("--mass", type=float, default=0.15, help="dynamic mode: bar mass (kg)")
    ap.add_argument("--friction", type=float, default=1.0, help="bar friction coefficient")
    ap.add_argument("--friction-combine", choices=["average", "min", "multiply", "max"], default="average",
                    help="how the bar's friction mixes with the robot's; 'min' makes --friction an upper limit")
    ap.add_argument("--closure", type=float, default=1.0,
                    help="final hand closure, 0 to 1. 0 is a control: the bar must fall")
    ap.add_argument("--close-time", type=float, default=2.0, help="seconds from open to fully closed")
    ap.add_argument("--max-force", type=float, default=30.0, help="drive limit for all joints")
    ap.add_argument("--finger-limit", type=float, default=None,
                    help="torque limit for the 20 hand joints only (N*m); uses the collider-only robot build")
    ap.add_argument("--finger-stiffness", type=float, default=20.0, help="used with --finger-limit (N*m/rad)")
    ap.add_argument("--finger-damping", type=float, default=0.1, help="used with --finger-limit (N*m*s/rad)")
    args = ap.parse_args()

    urdf = os.path.expanduser(args.urdf)
    world = make_world()
    finger_drive = None
    if args.finger_limit is not None:
        finger_drive = (args.finger_stiffness, args.finger_damping, args.finger_limit)
    robot = ICub(world, urdf, base_position=BASE_POSITION, max_force=args.max_force, finger_drive=finger_drive)

    # The URDF hand frame relative to the PhysX COM frame is fixed, so measure it once at the zero pose.
    world.step(1e-4)
    root = ET.parse(urdf).getroot()
    hand_link = robot.link("r_hand")
    T_com_to_urdf = np.linalg.inv(link_pose(hand_link)) @ urdf_link_frames(root, BASE_POSITION)["r_hand"]

    # One contact sensor on every hand part (self-collision is off, so they only feel the bar).
    sensors = {}
    for name in PARTS:
        sensors[name] = tp.ContactSensor(robot.collider(name))
        world.register_sensor(sensors[name])

    with open(POSES) as f:
        pose = yaml.safe_load(f)["poses"]["reach_forward"]
    robot.set_targets({j: math.radians(v) for j, v in pose.items()})
    robot.set_targets(robot.hand_targets(0.0))
    robot.apply()
    if args.mode == "dynamic":
        world.set_gravity(tp.Vector3(0, 0, 0))
    for _ in range(int(SETTLE_S / DT)):
        world.step(DT)
    print(f"arm settled: worst joint err {math.degrees(robot.tracking_error()):.2f} deg")

    T_hand = link_pose(hand_link) @ T_com_to_urdf
    T_bar_in_hand = np.eye(4)
    T_bar_in_hand[:3, 3] = np.array(args.pos) / 1000.0
    T_bar = T_hand @ T_bar_in_hand
    size, length = args.size / 1000.0, args.length / 1000.0
    bar_mesh = make_bar(T_bar, size, length)
    material = world.create_material(args.friction, args.friction, 0.0, friction_combine=args.friction_combine)
    bar_body = None
    if args.mode == "static":
        world.add_static(bar_mesh, material=material)
    else:
        bar_body = world.add(bar_mesh, density=args.mass / (size * size * length), material=material)

    def bar_in_hand_mm():
        hand = link_pose(hand_link) @ T_com_to_urdf
        p = bar_mesh.position
        return (np.linalg.inv(hand) @ np.array([p.x, p.y, p.z, 1.0]))[:3] * 1000.0

    first_touch = {}
    peak, hold_sum, hold_n = {n: 0.0 for n in PARTS}, {n: 0.0 for n in PARTS}, 0
    closure = 0.0

    def tick(in_hold):
        nonlocal hold_n
        robot.set_targets(robot.hand_targets(closure))
        robot.apply()
        world.step(DT)
        for name, sensor in sensors.items():
            samples = sensor.drain()
            force = 0.0
            for s in samples:
                v = s.force
                force = max(force, math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z))
            if force > 0.0 and name not in first_touch:
                first_touch[name] = closure
            peak[name] = max(peak[name], force)
            if in_hold:
                hold_sum[name] += force
        if in_hold:
            hold_n += 1

    # Close the hand slowly. The bar must not touch anything while the hand is fully open.
    for _ in range(5):
        tick(False)
    touching_open = sorted(first_touch)
    for step in range(int(args.close_time / DT)):
        closure = args.closure * (step + 1) * DT / args.close_time
        tick(False)
    closure = args.closure
    for _ in range(int(HOLD_S / DT)):
        tick(True)
    after_close = bar_in_hand_mm()

    print(f"bar: {args.size:.0f} x {args.size:.0f} x {args.length:.0f} mm, centre {args.pos} mm in the hand frame, "
          f"{args.mode}, friction {args.friction}, drive limit {args.max_force} N*m")
    print(f"touching while fully open: {touching_open or 'nothing'}\n")
    print(f"{'part':<20}{'first touch at closure':>24}{'peak force (N)':>16}{'hold force (N)':>16}")
    for name in PARTS:
        if name in first_touch or peak[name] > 0.0:
            print(f"{name:<20}{first_touch.get(name, float('nan')):>24.2f}{peak[name]:>16.1f}"
                  f"{hold_sum[name] / max(hold_n, 1):>16.1f}")
    missing = [f for f in FINGERS if not any(n in first_touch for n in PARTS if f"_{f}_" in n)]
    print(f"\nfingers with no contact at all: {missing or 'none'}")
    print(f"palm contact: {'yes' if 'r_hand' in first_touch else 'no'}")

    if args.mode == "dynamic":
        start = np.array(args.pos, dtype=float)
        print(f"\nbar moved {np.linalg.norm(after_close - start):.1f} mm while closing, "
              f"now at {np.round(after_close, 1)} mm in the hand frame")
        hand_R = (link_pose(hand_link) @ T_com_to_urdf)[:3, :3]
        print(f"gravity direction in the hand frame: {np.round(hand_R.T @ np.array([0.0, 0.0, -1.0]), 2)} "
              f"(the bar's long axis is z)")
        world.set_gravity(tp.Vector3(0, 0, -9.81))
        for _ in range(int(GRAVITY_ON_S / DT)):
            tick(False)
        final = bar_in_hand_mm()
        slip = float(np.linalg.norm(final - after_close))
        held = slip < 10.0
        print(f"gravity on for {GRAVITY_ON_S:.0f} s: bar slipped {slip:.1f} mm, now at {np.round(final, 1)} mm")
        print(f"RESULT: {'HELD' if held else 'NOT HELD'} (held means under 10 mm of slip)")


if __name__ == "__main__":
    main()
