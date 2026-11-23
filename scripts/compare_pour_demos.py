#!/usr/bin/env python3
"""Compare the expert's pours in two groups of recorded demos, split by the can's yaw.

Why: ACT trained on dataset v2 pours beside the pot when the can is turned the other way (negative yaw), and v1 does not.
If the new demos pour differently, or more variably, at negative yaw, the policy could average two ways of pouring.

For each demo, from the measured joints (observation state) and forward kinematics:
  - can yaw: the first number the seed's random generator draws (as in state_machine.run_episode)
  - grasp: the first frame with the thumb closed. The can is still upright on the table, so the can's "up"
    direction in the hand's frame is the world z axis seen from the hand at that frame
  - tilt(t): the angle between the can's up direction and the world z axis
  - pour: the frames with tilt over 20 deg. Reported: the heading the can tips towards at its furthest tilt,
    its furthest tilt, the hand position when the pour starts, the arm joints at the furthest tilt, and its length
Groups by seed: v1 = 1000-1203, new = 1204 and up. Split by yaw: below -10, -10..0, 0..10, over 10 deg.

Run in the gardener env:
  PYTHONPATH=. python scripts/compare_pour_demos.py [--raw data/raw] [--out data/eval/pour_demos.npz]
"""
import argparse
import glob
import os

import numpy as np

from gardener.expert.planner import PLAN_JOINTS
from gardener.expert.state_machine import BASE_POSITION
from gardener.sim.kinematics import ArmKinematics
from scripts.run_expert import DEFAULT_URDF

BINS = [(-25, -10), (-10, 0), (0, 10), (10, 25)]
TILT_DEG = 20.0


def one_demo(kin, path):
    z = np.load(path, allow_pickle=True)
    seed = int(z["seed"])
    state = z["state"]
    fps = float(z["fps"])
    hands = [kin.hand(dict(zip(PLAN_JOINTS, s[:len(PLAN_JOINTS)]))) for s in state]
    grasp = int(np.argmax(state[:, -1] > 0.5))
    up_in_hand = hands[grasp][:3, :3].T @ np.array([0.0, 0.0, 1.0])
    ups = np.array([T[:3, :3] @ up_in_hand for T in hands])
    tilt = np.degrees(np.arccos(np.clip(ups[:, 2], -1, 1)))
    pour = np.flatnonzero(tilt > TILT_DEG)
    top = int(np.argmax(tilt))
    return dict(seed=seed, yaw=np.random.default_rng(seed).uniform(-25, 25), text=str(z["instruction"]),
                heading=np.degrees(np.arctan2(ups[top, 1], ups[top, 0])), max_tilt=tilt[top],
                start_xyz=hands[pour[0]][:3, 3] * 1000 if len(pour) else np.full(3, np.nan),
                joints_top=np.degrees(state[top, :len(PLAN_JOINTS)]), pour_s=len(pour) / fps,
                pour_start_s=pour[0] / fps if len(pour) else np.nan)


def summary(rows, name):
    if not rows:
        print(f"  {name}: none")
        return
    h = np.array([r["heading"] for r in rows])
    j = np.array([r["joints_top"] for r in rows])
    xyz = np.array([r["start_xyz"] for r in rows])
    headings, counts = np.unique(np.round(h / 15) * 15, return_counts=True)
    modes = ", ".join(f"{int(a)}:{c}" for a, c in zip(headings, counts))
    print(f"  {name:4} n={len(rows):3} | pour heading deg mean {h.mean():6.1f} sd {h.std():5.1f} (15-deg bins {modes})")
    print(f"       max tilt {np.mean([r['max_tilt'] for r in rows]):5.1f} | pour length s {np.mean([r['pour_s'] for r in rows]):4.1f}"
          f" | pour starts at s {np.nanmean([r['pour_start_s'] for r in rows]):5.1f}"
          f" | hand at start mm sd x/y/z {np.nanstd(xyz, 0).round(0)}")
    print(f"       joints at furthest tilt, sd deg: {dict(zip(PLAN_JOINTS, j.std(0).round(1)))}")
    print(f"       joints at furthest tilt, mean deg: {dict(zip(PLAN_JOINTS, j.mean(0).round(1)))}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/eval/pour_demos.npz")
    ap.add_argument("--urdf", default=DEFAULT_URDF)
    args = ap.parse_args()
    kin = ArmKinematics(os.path.expanduser(args.urdf), base=BASE_POSITION)
    rows = [one_demo(kin, p) for p in sorted(glob.glob(os.path.join(args.raw, "episode_*.npz")))]
    groups = {"v1": [r for r in rows if r["seed"] < 1204], "new": [r for r in rows if r["seed"] >= 1204]}
    for name, g in groups.items():
        y = np.array([r["yaw"] for r in g])
        print(f"{name}: {len(g)} demos; can yaw per bin {dict(zip([f'{a}..{b}' for a, b in BINS], np.histogram(y, [b[0] for b in BINS] + [25])[0]))};"
              f" mean {y.mean():+.1f}, sd {y.std():.1f}")
    for a, b in BINS:
        print(f"\ncan yaw {a}..{b} deg")
        for name, g in groups.items():
            summary([r for r in g if a <= r["yaw"] < b], name)
    np.savez(args.out, **{k: np.array([r[k] for r in rows]) for k in rows[0]})
    print(f"\nper-demo table: {args.out}")


if __name__ == "__main__":
    main()
