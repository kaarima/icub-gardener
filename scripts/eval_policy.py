#!/usr/bin/env python3
"""Evaluate a trained policy on the 50 held-out seeds, with exactly the expert's report.

The policy replaces the expert after the scene is set up: same scenes, same contact checks, same success
rule (gardener/expert/state_machine.py, run_controller and score). Each episode starts its own policy server
in the gardener-train env. Two workers fit the 4 GB laptop GPU.

Usage (conda env gardener):
  python scripts/eval_policy.py outputs/act/checkpoints/last/pretrained_model | tee data/eval/act_heldout.txt
  python scripts/eval_policy.py CKPT --ensemble 0.01        # ask every frame and average overlapping chunks
  python scripts/eval_policy.py CKPT --first-seed 0         # the dev seeds instead
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_expert  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1].startswith("-"):
        raise SystemExit(__doc__)
    args = ["--policy", sys.argv[1]] + sys.argv[2:]
    if "--first-seed" not in args:
        args += ["--first-seed", "5000"]
    if "--workers" not in args:
        args += ["--workers", "2"]
    sys.argv = [sys.argv[0]] + args
    eval_expert.main()
