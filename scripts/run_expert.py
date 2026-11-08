#!/usr/bin/env python3
"""Run the scripted gardener for one seed and instruction, and report the book's metrics.

Usage:
  python scripts/run_expert.py --seed 0 --instruction "Water the left plant."
  python scripts/run_expert.py --seed 3 --instruction "Give the right plant a lot of water."

Nothing is written to disk.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gardener.expert.state_machine import run_episode  # noqa: E402

DEFAULT_URDF = "~/icub-assets/iCub/robots/iCubGazeboV2_10/gardener_hand.urdf"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--urdf", default=DEFAULT_URDF)
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "scene.yaml"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--instruction", default="Water the left plant.")
    ap.add_argument("--can-yaw", type=float, default=None, help="deg; overrides the seed's random rotation")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    r = run_episode(os.path.expanduser(args.urdf), args.config, os.path.join(ROOT, "configs", "poses.yaml"),
                    args.instruction, args.seed, verbose=not args.quiet, can_yaw_deg=args.can_yaw)
    print(f"RESULT: {'SUCCESS' if r.success else 'FAILED: ' + r.failure}, reached {r.stage_reached}")
    print(f"water: poured {r.poured_fraction * 100:.0f}% of the can, {r.water_in_target * 100:.0f}% of that in the target pot, "
          f"spill {r.spill * 100:.0f}%, wrong pot {r.wrong_pot}, amount in band {r.amount_ok}, counts {r.counts}")
    print(f"can: drag {r.drag_mm:.1f} mm, lift {r.lift_mm:.1f} mm, final tilt {r.final_tilt_deg:.1f} deg, placed {r.placed_error_mm:.0f} mm from the start")
    print(f"wrist_yaw margin {r.min_wrist_yaw_margin_deg:.1f} deg; {r.sim_time:.1f} s simulated in {r.wall_time:.1f} s = {r.sim_time / max(r.wall_time, 1e-6):.2f} x real time")


if __name__ == "__main__":
    main()
