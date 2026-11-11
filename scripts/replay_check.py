#!/usr/bin/env python3
"""Play recorded episodes' saved actions back in the simulator, open loop, and compare with the recording.

This is the best a policy that copied the demonstrations perfectly could do: same scene, one action per frame,
no expert feedback. Each replay must succeed, pour about the same amount into the same pot, and keep the joints
close to the recorded ones. Picks episodes at random (fixed seed) unless given files.

Usage (conda env gardener):
  python scripts/replay_check.py --count 5
"""
import argparse
import glob
import os
import random
import re
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*")
    ap.add_argument("--raw", default=os.path.join(ROOT, "data", "raw"))
    ap.add_argument("--count", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    files = args.files or sorted(random.Random(args.seed).sample(sorted(glob.glob(os.path.join(args.raw, "episode_*.npz"))), args.count))
    rows, ok = [], 0
    for path in files:
        e = np.load(path)
        text, seed = str(e["instruction"]), int(e["seed"])
        p = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "run_expert.py"), "--seed", str(seed), "--instruction", text,
                            "--quiet", "--replay", path], capture_output=True, text=True, timeout=1200)
        o = p.stdout + p.stderr
        res = re.search(r"RESULT: (SUCCESS|FAILED: [^,]*)", o)
        water = re.search(r"poured (\d+)% of the can, (\d+)% of that in the target pot, spill (\d+)%", o)
        dev = re.search(r"largest ([0-9.]+) deg, median per frame ([0-9.]+) deg", o)
        contacts = re.search(r"contacts: (.*)", o)
        if not (res and water and dev):
            rows.append(f"{os.path.basename(path)}: CRASH {o[-300:]}")
            continue
        good = res[1] == "SUCCESS"
        ok += good
        rows.append(f"{os.path.basename(path)} seed {seed} {text!r}: {res[1]}; poured {water[1]}% (recorded {float(e['meta_poured']) * 100:.0f}%), "
                    f"in target {water[2]}% (recorded {float(e['meta_water_in_target']) * 100:.0f}%), spill {water[3]}%; "
                    f"joints within {dev[1]} deg (median {dev[2]}); contacts {contacts[1] if contacts else '?'}")
        print(rows[-1], flush=True)
    print(f"\nREPLAY: {ok}/{len(files)} succeed")


if __name__ == "__main__":
    main()
