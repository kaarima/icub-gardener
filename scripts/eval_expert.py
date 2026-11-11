#!/usr/bin/env python3
"""Evaluate the scripted gardener on N randomised scenes (the book's objective O2: at least 80% over 50).

Each episode runs in its own process, because PhysX allows one world per process. Instructions are
balanced across target pot and amount. Prints every episode, then the rates with a 95% interval.
A failed episode lists its first failure reason and all of them: one episode can miss the pot AND touch a rim.

Usage:
  python scripts/eval_expert.py --episodes 50 --workers 4
  python scripts/eval_expert.py --from-report docs/eval/act_heldout.txt   # rebuild a saved report, no simulation

Nothing is written to disk.
"""
import argparse
import concurrent.futures as cf
import math
import os
import re
import subprocess
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSTRUCTIONS = ["Water the left plant.", "Water the right plant.",
                "Give the left plant just a little water.", "Give the right plant just a little water.",
                "Give the left plant a lot of water.", "Give the right plant a lot of water."]
REASONS = ["nothing poured", "wrong pot", "water missed the pot", "can not upright on the table", "timeout", "collision"]


EXTRA = []   # options passed on to run_expert.py, such as --policy


def run(job):
    seed, text = job
    p = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "run_expert.py"), "--seed", str(seed), "--instruction", text] + EXTRA,
                       capture_output=True, text=True, timeout=1800)
    o = p.stdout + p.stderr
    result = re.search(r"RESULT: (SUCCESS|FAILED: [^,]*), reached (.*)", o)
    if not result:
        return dict(seed=seed, text=text, ok=False, failure="crash", failures=["crash"], stage="?", raw=o[-300:])
    water = re.search(r"poured (\d+)% of the can, (\d+)% of that in the target pot, spill (\d+)%, wrong pot (\w+), amount in band (\w+)", o)
    can = re.search(r"drag ([0-9.]+) mm, lift ([-0-9.]+) mm, final tilt ([0-9.]+) deg, placed (\d+) mm", o)
    wrist = re.search(r"wrist_yaw margin ([-0-9.]+) deg; ([0-9.]+) s simulated in ([0-9.]+) s", o)
    spout = re.search(r"spout ([0-9.]+) mm off target, (\d+)% lost", o)
    contact = re.search(r"layouts redrawn (\d+); contacts: (.*)", o)
    jump = re.search(r"largest jump ([0-9.]+) deg/frame; return starts ([0-9.]+) deg, ([0-9.]+) mm from the arm's pose; worst in (.*)", o)
    clear = dict((k, float(v)) for k, v in re.findall(r"([a-z ]+) (-?[0-9.]+)(?:,|$)", (re.search(r"rim clearance mm: (.*)", o) or [0, ""])[1]))
    height = re.search(r"pour's start: ([-+0-9.]+) mm", o)
    reasons = re.search(r"failure reasons: (.*)", o)
    setup = re.search(r"can yaw ([-+0-9.]+) deg, shift \[\s*([-0-9.]+)\s+([-0-9.]+)\s*\] mm, (\d+)% full", o)
    failure = "" if result[1] == "SUCCESS" else result[1][8:]
    return dict(seed=seed, text=text, ok=result[1] == "SUCCESS", failure=failure, stage=result[2],
                failures=[] if not failure else reasons[1].split("; ") if reasons else [failure],
                poured=int(water[1]), target=int(water[2]), spill=int(water[3]), wrong=water[4] == "True", amount=water[5] == "True",
                drag=float(can[1]), lift=float(can[2]), tilt=float(can[3]), placed=int(can[4]), margin=float(wrist[1]),
                sim=float(wrist[2]), wall=float(wrist[3]), spout=float(spout[1]), lost=int(spout[2]), redrawn=int(contact[1]), contacts=contact[2], yaw=float(setup[1]), sx=float(setup[2]), sy=float(setup[3]), fill=int(setup[4]),
                jump=float(jump[1]), gap=float(jump[2]), gap_mm=float(jump[3]), jump_phase=jump[4], clear={k.strip(): v for k, v in clear.items()}, height=float(height[1]))


def wilson(k, n):
    """95% confidence interval for a success rate."""
    if n == 0:
        return 0.0, 0.0
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def read_report(path):
    """Rebuild the per-episode results from a report this script printed earlier, with no simulation.

    A report saved before the "all reasons" column existed only shows the first reason, so every reason is
    worked out again from the printed numbers, with the rules of score() in gardener/expert/state_machine.py:
    nothing poured (pour% is 0), wrong pot (more in the other pot than in the target: 100 - tgt% - spill > tgt%),
    water missed (tgt% under the threshold), can not upright (tilt over the limit; the 1 cm height check is not
    printed, so it is left out), timeout (stopped by the time limit), collision (a "contacts:" line). Rounded
    percentages can put a borderline episode on the other side of a threshold, so the first reason found is
    compared with the one the report printed. Returns (results, header lines, lines that cannot be recomputed).
    """
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "scene.yaml")))["success"]
    lines = open(path, encoding="utf-8").read().splitlines()
    res, header, kept, contacts, clear, redrawn = [], [], [], {}, {}, 0
    row = re.compile(r"^\s*(\d+) (.{42})(ok|FAIL|CRASH)\s*(.*)$")
    for line in lines:
        m = row.match(line)
        if m and not line.startswith("seed"):
            seed, text, result, rest = int(m[1]), m[2].strip(), m[3], m[4]
            if result == "CRASH":
                res.append(dict(seed=seed, text=text, ok=False, failure="crash", failures=["crash"], stage="?", raw=rest))
                continue
            cols = re.match(r"((?:\s*\S+){16})(.*)$", rest)      # 16 number columns, then the failure text
            f, tail = cols[1].split(), cols[2]
            r = dict(seed=seed, text=text, ok=result == "ok", yaw=float(f[0]), sx=float(f[1]), sy=float(f[2]), fill=int(f[3]),
                     drag=float(f[4]), lift=float(f[5]), poured=int(f[6]), target=int(f[7]), spill=int(f[8]), jump=float(f[9]),
                     amount=f[10] == "yes", tilt=float(f[11]), placed=int(f[12]), margin=float(f[13]), spout=float(f[14]),
                     lost=int(f[15]), contacts="none", redrawn=0, clear={}, gap=0.0, gap_mm=0.0, jump_phase="")
            t = re.match(r"(.*?) \(reached (.*?)\)", tail.strip())
            r["failure"], r["stage"] = (t[1], t[2]) if t else ("", "end")
            r["wrong"] = r["poured"] > 0 and 100 - r["target"] - r["spill"] > r["target"]
            res.append(r)
            continue
        if m := re.match(r"seed (\d+) contacts: (.*)", line):
            contacts[int(m[1])] = m[2]
        elif m := re.match(r"scene layouts redrawn because the can overlapped a pot: (\d+)", line):
            redrawn = int(m[1])
        elif m := re.match(r"\s*(\d+):\s+([0-9.]+) (.*?)\s*\|\s+([0-9.]+)\s+([0-9.]+) \|(.*)", line):
            clear[int(m[1])] = (float(m[2]), m[3], float(m[4]), float(m[5]),
                                {k: float(v) for k, v in re.findall(r"(\w[\w ]*?)=(-?[0-9.]+)", m[6])})
        elif line.startswith(("POLICY:", "evaluating the policy")):
            header.append(line)
        elif line.startswith(("speed:", "can height at the pour's start")):
            kept.append(line)
    for r in res:
        if "drag" not in r:
            continue
        r["contacts"] = contacts.get(r["seed"], "none")
        if r["seed"] in clear:
            r["jump"], r["jump_phase"], r["gap"], r["gap_mm"], r["clear"] = clear[r["seed"]]
        checks = [("nothing poured", r["poured"] == 0), ("wrong pot", r["wrong"]),
                  ("water missed the pot", r["poured"] > 0 and r["target"] < 100 * cfg["min_water_in_target"]),
                  ("can not upright on the table", r["tilt"] > cfg["max_can_tilt"]),
                  ("timeout", r["stage"] == "time limit" or r["failure"] == "timeout"),
                  ("collision", r["contacts"] != "none")]
        found = [name for name, failed in checks if failed]
        if r["failure"] and r["failure"] not in REASONS:
            found = [r["failure"]] + found          # an early stop, such as "no place plan", comes first
        if r["failure"] and r["failure"] not in found:
            found = [r["failure"]] + found          # trust the printed reason when a rounded number hides it
        r["failures"] = found if not r["ok"] else []
        if r["ok"] and found:
            print(f"note: seed {r['seed']} succeeded but the printed numbers suggest {found}: rounding, kept as a success")
        elif r["failure"] and found[0] != r["failure"]:
            print(f"note: seed {r['seed']}: printed first reason {r['failure']!r}, recomputed {found[0]!r}; the printed one is kept")
            r["failures"] = [r["failure"]] + [x for x in found if x != r["failure"]]
    if res:
        res[0]["redrawn"] = redrawn
    return res, header, kept


def report(res, workers, kept=()):
    """Print the per-episode table and the summary. kept: lines from a saved report that cannot be recomputed."""
    print(f"{'seed':>4} {'instruction':<42}{'result':<8}{'yaw':>6}{'dx':>5}{'dy':>5}{'fill':>5}{'drag':>6}{'lift':>6}{'pour%':>6}{'tgt%':>5}{'spill':>6}{'jump':>6}{'band':>5}{'tilt':>5}{'place':>6}{'marg':>6}{'spout':>6}{'lost':>5}  first reason (reached) | all reasons")
    for r in res:
        if "drag" not in r:
            print(f"{r['seed']:>4} {r['text']:<42}CRASH   {r.get('raw', '')}")
            continue
        line = (f"{r['seed']:>4} {r['text']:<42}{'ok' if r['ok'] else 'FAIL':<8}{r['yaw']:6.1f}{r['sx']:5.0f}{r['sy']:5.0f}{r['fill']:5d}{r['drag']:6.1f}{r['lift']:6.1f}"
                f"{r['poured']:6d}{r['target']:5d}{r['spill']:6d}{r['jump']:6.1f}{'yes' if r['amount'] else 'no':>5}{r['tilt']:5.1f}{r['placed']:6d}{r['margin']:6.1f}{r['spout']:6.1f}{r['lost']:5d}")
        if not r["ok"]:
            line += f"  {r['failure']} (reached {r['stage']}) | {'; '.join(r['failures'])}"
        print(line)
    n = len(res)
    k = sum(r["ok"] for r in res)
    lo, hi = wilson(k, n)
    done = [r for r in res if "drag" in r]
    for r in done:
        if r["contacts"] != "none":
            print(f"seed {r['seed']} contacts: {r['contacts']}")
    print(f"\nscene layouts redrawn because the can overlapped a pot: {sum(r['redrawn'] for r in done)} over {len(done)} scenes")
    print(f"\nSUCCESS: {k}/{n} = {100 * k / n:.0f}% (95% interval {100 * lo:.0f}-{100 * hi:.0f}%)")
    ok = [r for r in done if r["ok"]]
    if ok:
        def rng(key):
            v = sorted(r[key] for r in ok)
            return f"median {v[len(v) // 2]:.1f}, range {v[0]:.1f}-{v[-1]:.1f}"
        print(f"successful episodes: water in target % {rng('target')}; drag mm {rng('drag')}; lift mm {rng('lift')}; wrist margin deg {rng('margin')}")
        print(f"amount in the instruction's band: {sum(r['amount'] for r in ok)}/{len(ok)}")
        if all("sim" in r for r in ok):
            print(f"speed: {sum(r['sim'] for r in ok) / sum(r['wall'] for r in ok):.2f} x real time with {workers} workers; mean episode {sum(r['sim'] for r in ok) / len(ok):.1f} s simulated")
        else:
            print(*[f"{x}   (from the saved report)" for x in kept if x.startswith("speed:")], sep="\n")
    if done:
        spills = sorted(r["spill"] for r in done)
        worst = max(done, key=lambda r: r["spill"])
        print(f"spill, % of the poured water on the table or floor: median {spills[len(spills) // 2]}, worst {spills[-1]} (seed {worst['seed']}); over 5%: {sum(v > 5 for v in spills)}/{len(spills)}")
        jumps = sorted(r["jump"] for r in done)
        print(f"largest joint target jump, deg/frame: median {jumps[len(jumps) // 2]:.2f}, worst {jumps[-1]:.2f}; over 2: {sum(v > 2 for v in jumps)}/{len(jumps)}")
        print("\nseed: jump phase | return gap deg, mm | rim clearance mm per phase")
        for r in done:
            print(f"{r['seed']:>4}: {r['jump']:5.1f} {r['jump_phase']:<10}| {r['gap']:5.1f} {r['gap_mm']:5.1f} | " + " ".join(f"{k}={v:.1f}" for k, v in r["clear"].items()))
        if all("height" in r for r in done):
            heights = sorted(r["height"] for r in done)
            print(f"can height at the pour's start, mm from the plan: lowest {heights[0]:+.1f}, median {heights[len(heights) // 2]:+.1f}, highest {heights[-1]:+.1f}")
        else:
            print(*[f"{x}   (from the saved report)" for x in kept if x.startswith("can height")], sep="\n")
        for phase in ("transport", "pour", "tilt back", "return"):
            v = sorted(r["clear"][phase] for r in done if phase in r["clear"])
            if v:
                print(f"rim clearance in {phase}, mm: lowest {v[0]:.1f}, median {v[len(v) // 2]:.1f}, under 5: {sum(x < 5 for x in v)}/{len(v)}")
    print("\nby instruction:")
    for text in INSTRUCTIONS:
        mine = [r for r in done if r["text"] == text]
        if mine:
            print(f"  {text:<42} success {sum(r['ok'] for r in mine)}/{len(mine)}, amount in band {sum(r['amount'] for r in mine)}/{len(mine)}, "
                  f"wrong pot {sum(r['wrong'] for r in mine)}/{len(mine)}")
    first, every = {}, {}
    for r in res:
        if not r["ok"]:
            first[r["failure"]] = first.get(r["failure"], 0) + 1
            for x in r["failures"]:
                every[x] = every.get(x, 0) + 1
    print("failures by type:", first or "none")
    print("episodes with each failure reason (one episode can have several):", every or "none")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--policy", default=None, help="evaluate this trained policy (a pretrained_model folder) instead of the expert")
    ap.add_argument("--n-action-steps", type=int, default=None)
    ap.add_argument("--ensemble", type=float, default=None)
    ap.add_argument("--from-report", default=None, help="rebuild the report from this saved one, with all failure reasons; runs nothing")
    args = ap.parse_args()
    if args.from_report:
        res, header, kept = read_report(args.from_report)
        for line in header:
            print(line)
        print(f"rebuilt from {args.from_report}, no simulation run")
        report(res, args.workers, kept)
        return
    if args.policy:
        EXTRA.extend(["--policy", os.path.abspath(args.policy)])
        if args.n_action_steps is not None:
            EXTRA.extend(["--n-action-steps", str(args.n_action_steps)])
        if args.ensemble is not None:
            EXTRA.extend(["--ensemble", str(args.ensemble)])
        print(f"evaluating the policy {args.policy}, seeds {args.first_seed}-{args.first_seed + args.episodes - 1}")
    jobs = [(args.first_seed + i, INSTRUCTIONS[i % len(INSTRUCTIONS)]) for i in range(args.episodes)]
    with cf.ThreadPoolExecutor(args.workers) as ex:
        res = list(ex.map(run, jobs))
    report(res, args.workers)


if __name__ == "__main__":
    main()
