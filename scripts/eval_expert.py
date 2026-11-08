#!/usr/bin/env python3
"""Evaluate the scripted gardener on N randomised scenes (the book's objective O2: at least 80% over 50).

Each episode runs in its own process, because PhysX allows one world per process. Instructions are
balanced across target pot and amount. Prints every episode, then the rates with a 95% interval.

Usage:
  python scripts/eval_expert.py --episodes 50 --workers 4

Nothing is written to disk.
"""
import argparse
import concurrent.futures as cf
import math
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSTRUCTIONS = ["Water the left plant.", "Water the right plant.",
                "Give the left plant just a little water.", "Give the right plant just a little water.",
                "Give the left plant a lot of water.", "Give the right plant a lot of water."]


def run(job):
    seed, text = job
    p = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "run_expert.py"), "--seed", str(seed), "--instruction", text],
                       capture_output=True, text=True, timeout=900)
    o = p.stdout + p.stderr
    result = re.search(r"RESULT: (SUCCESS|FAILED: [^,]*), reached (.*)", o)
    if not result:
        return dict(seed=seed, text=text, ok=False, failure="crash", stage="?", raw=o[-300:])
    water = re.search(r"poured (\d+)% of the can, (\d+)% of that in the target pot, spill (\d+)%, wrong pot (\w+), amount in band (\w+)", o)
    can = re.search(r"drag ([0-9.]+) mm, lift ([-0-9.]+) mm, final tilt ([0-9.]+) deg, placed (\d+) mm", o)
    wrist = re.search(r"wrist_yaw margin ([-0-9.]+) deg; ([0-9.]+) s simulated in ([0-9.]+) s", o)
    setup = re.search(r"can yaw ([-+0-9.]+) deg, shift \[\s*([-0-9.]+)\s+([-0-9.]+)\s*\] mm, (\d+)% full", o)
    return dict(seed=seed, text=text, ok=result[1] == "SUCCESS", failure="" if result[1] == "SUCCESS" else result[1][8:], stage=result[2],
                poured=int(water[1]), target=int(water[2]), spill=int(water[3]), wrong=water[4] == "True", amount=water[5] == "True",
                drag=float(can[1]), lift=float(can[2]), tilt=float(can[3]), placed=int(can[4]), margin=float(wrist[1]),
                sim=float(wrist[2]), wall=float(wrist[3]), yaw=float(setup[1]), sx=float(setup[2]), sy=float(setup[3]), fill=int(setup[4]))


def wilson(k, n):
    """95% confidence interval for a success rate."""
    if n == 0:
        return 0.0, 0.0
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    jobs = [(args.first_seed + i, INSTRUCTIONS[i % len(INSTRUCTIONS)]) for i in range(args.episodes)]
    with cf.ThreadPoolExecutor(args.workers) as ex:
        res = list(ex.map(run, jobs))

    print(f"{'seed':>4} {'instruction':<42}{'result':<8}{'yaw':>6}{'dx':>5}{'dy':>5}{'fill':>5}{'drag':>6}{'lift':>6}{'pour%':>6}{'tgt%':>5}{'band':>5}{'tilt':>5}{'place':>6}{'marg':>6}  failure")
    for r in res:
        if "drag" not in r:
            print(f"{r['seed']:>4} {r['text']:<42}CRASH   {r.get('raw', '')}")
            continue
        print(f"{r['seed']:>4} {r['text']:<42}{'ok' if r['ok'] else 'FAIL':<8}{r['yaw']:6.1f}{r['sx']:5.0f}{r['sy']:5.0f}{r['fill']:5d}{r['drag']:6.1f}{r['lift']:6.1f}"
              f"{r['poured']:6d}{r['target']:5d}{'yes' if r['amount'] else 'no':>5}{r['tilt']:5.1f}{r['placed']:6d}{r['margin']:6.1f}  {r['failure']} (reached {r['stage']})" if not r['ok'] else
              f"{r['seed']:>4} {r['text']:<42}{'ok':<8}{r['yaw']:6.1f}{r['sx']:5.0f}{r['sy']:5.0f}{r['fill']:5d}{r['drag']:6.1f}{r['lift']:6.1f}"
              f"{r['poured']:6d}{r['target']:5d}{'yes' if r['amount'] else 'no':>5}{r['tilt']:5.1f}{r['placed']:6d}{r['margin']:6.1f}")
    n = len(res)
    k = sum(r["ok"] for r in res)
    lo, hi = wilson(k, n)
    done = [r for r in res if "drag" in r]
    print(f"\nSUCCESS: {k}/{n} = {100 * k / n:.0f}% (95% interval {100 * lo:.0f}-{100 * hi:.0f}%)")
    ok = [r for r in done if r["ok"]]
    if ok:
        def rng(key):
            v = sorted(r[key] for r in ok)
            return f"median {v[len(v) // 2]:.1f}, range {v[0]:.1f}-{v[-1]:.1f}"
        print(f"successful episodes: water in target % {rng('target')}; drag mm {rng('drag')}; lift mm {rng('lift')}; wrist margin deg {rng('margin')}")
        print(f"amount in the instruction's band: {sum(r['amount'] for r in ok)}/{len(ok)}")
        print(f"speed: {sum(r['sim'] for r in ok) / sum(r['wall'] for r in ok):.2f} x real time with {args.workers} workers; mean episode {sum(r['sim'] for r in ok) / len(ok):.1f} s simulated")
    failures = {}
    for r in res:
        if not r["ok"]:
            failures[r["failure"]] = failures.get(r["failure"], 0) + 1
    print("failures by type:", failures or "none")


if __name__ == "__main__":
    main()
