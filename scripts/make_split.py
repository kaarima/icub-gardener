#!/usr/bin/env python3
"""Split the dataset's episodes into training and validation, balanced by instruction, and save the lists.

Validation episodes are never trained on: scripts/val_loss.py measures the policy's error on them, to see
whether it learns the task or only memorises its training episodes. Fixed seed, so the split is repeatable.

Run in the gardener-train env:
  env -u PYTHONPATH ~/miniconda3/envs/gardener-train/bin/python scripts/make_split.py --val 20
"""
import argparse
import json
import os
import random
from collections import defaultdict

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.join(ROOT, "data", "lerobot", "gardener_v1"))
    ap.add_argument("--repo-id", default="Kimaa6/icub_gardener_v1")
    ap.add_argument("--val", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "configs", "act_split.json"))
    args = ap.parse_args()
    meta = LeRobotDatasetMetadata(args.repo_id, root=args.root)
    by_task = defaultdict(list)
    for ep in range(meta.total_episodes):
        by_task[meta.episodes[ep]["tasks"][0]].append(ep)
    rng = random.Random(args.seed)
    tasks = sorted(by_task)
    val = []
    # Take episodes from each instruction in turn, so validation covers all of them evenly.
    pools = {t: rng.sample(by_task[t], len(by_task[t])) for t in tasks}
    while len(val) < args.val:
        for t in tasks:
            if pools[t] and len(val) < args.val:
                val.append(pools[t].pop())
    val = sorted(val)
    train = [ep for ep in range(meta.total_episodes) if ep not in set(val)]
    with open(args.out, "w") as f:
        json.dump({"dataset": args.repo_id, "seed": args.seed, "train": train, "val": val}, f, indent=1)
    print(f"{len(train)} training and {len(val)} validation episodes, saved to {args.out}")
    for t in tasks:
        print(f"  {t!r}: {sum(ep in val for ep in by_task[t])} val, {sum(ep not in val for ep in by_task[t])} train")


if __name__ == "__main__":
    main()
