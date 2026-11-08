#!/usr/bin/env python3
"""Run the expert over seeds and keep only successful episodes, as raw files for scripts/to_lerobot.py.

Each seed runs in its own process, because PhysX allows one world per process. Seeds already recorded or
already tried are skipped, so the script can be stopped and restarted without losing finished episodes.
A failed attempt is noted in data/raw/failed_seeds.txt and never saved. The instruction cycles with the
seed, so kept episodes are roughly, not exactly, balanced across target pot and amount.

Usage (conda env gardener):
  python scripts/record_episodes.py --count 20 --workers 2
"""
import argparse
import concurrent.futures as cf
import glob
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSTRUCTIONS = ["Water the left plant.", "Water the right plant.",
                "Give the left plant just a little water.", "Give the right plant just a little water.",
                "Give the left plant a lot of water.", "Give the right plant a lot of water."]


def attempt(job):
    seed, text, out = job
    t0 = time.time()
    p = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "run_expert.py"), "--seed", str(seed), "--instruction", text,
                        "--quiet", "--record", out], capture_output=True, text=True, timeout=1200)
    line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT")), p.stderr[-200:])
    return seed, text, os.path.exists(out), line, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", type=int, default=20, help="successful episodes wanted in total")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "raw"))
    ap.add_argument("--first-seed", type=int, default=1000, help="seeds 0-49 are kept for evaluation, as the book asks")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    failed_path = os.path.join(args.out, "failed_seeds.txt")
    failed = set()
    if os.path.exists(failed_path):
        with open(failed_path) as f:
            failed = {int(line.split()[0]) for line in f if line.strip()}

    def done():
        return sorted(glob.glob(os.path.join(args.out, "episode_*.npz")))

    def used(seed):
        return seed in failed or os.path.exists(os.path.join(args.out, f"episode_{seed:05d}.npz"))

    seed = args.first_seed
    tries = 0
    with cf.ThreadPoolExecutor(args.workers) as ex:
        while len(done()) < args.count:
            jobs = []
            while len(jobs) < min(args.workers, args.count - len(done())):
                if not used(seed):
                    # The instruction follows the seed, so a rerun gives the same scene and the same episode.
                    jobs.append((seed, INSTRUCTIONS[seed % len(INSTRUCTIONS)], os.path.join(args.out, f"episode_{seed:05d}.npz")))
                seed += 1
            for s, text, saved, line, wall in ex.map(attempt, jobs):
                tries += 1
                print(f"seed {s} {'saved' if saved else 'not saved'} ({wall:.0f} s): {text} | {line[:90]}", flush=True)
                if not saved:
                    failed.add(s)
                    with open(failed_path, "a") as f:
                        f.write(f"{s} {line[:90]}\n")
    print(f"{len(done())} episodes in {args.out}; {tries} attempts in this run, {len(failed)} failed seeds so far")


if __name__ == "__main__":
    main()
