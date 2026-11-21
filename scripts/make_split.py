#!/usr/bin/env python3
"""Split the dataset's episodes into training and validation, balanced by instruction, and save the lists.

Validation episodes are never trained on: scripts/val_loss.py measures the policy's error on them, to see
whether it learns the task or only memorises its training episodes. Fixed seed, so the split is repeatable.

With --keep-val, the validation episodes of an earlier split stay in validation, and only episodes that the
earlier split did not contain are drawn as new validation episodes. That keeps the old validation episodes
unseen in training when the dataset grows. --old-root checks first that the earlier dataset's episodes are the
same episodes, at the same indices, in this one (same instruction and frame count for each).

Run in the gardener-train env:
  env -u PYTHONPATH ~/miniconda3/envs/gardener-train/bin/python scripts/make_split.py --val 20
  # dataset v2 (400 episodes): the 20 v1 validation episodes plus 20 new ones, 40 of 400
  env -u PYTHONPATH ~/miniconda3/envs/gardener-train/bin/python scripts/make_split.py \\
      --root data/lerobot/gardener_v2 --repo-id Kimaa6/icub_gardener_v2 --val 40 \\
      --keep-val configs/act_split.json --old-root data/lerobot/gardener_v1 --out configs/act_split_v2.json
"""
import argparse
import json
import os
import random
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def choose_val(tasks, n_val, seed=0, keep=(), candidates=None):
    """Validation episodes: all of `keep`, then episodes from `candidates`, one instruction at a time.

    tasks[ep] is episode ep's instruction. Returns the sorted validation list.
    """
    candidates = range(len(tasks)) if candidates is None else candidates
    by_task = defaultdict(list)
    for ep in candidates:
        if ep not in keep:
            by_task[tasks[ep]].append(ep)
    rng = random.Random(seed)
    val = list(keep)
    # Take episodes from each instruction in turn, so validation covers all of them evenly.
    pools = {t: rng.sample(by_task[t], len(by_task[t])) for t in sorted(by_task)}
    while len(val) < n_val and any(pools.values()):
        for t in sorted(pools):
            if pools[t] and len(val) < n_val:
                val.append(pools[t].pop())
    if len(val) < n_val:
        raise SystemExit(f"only {len(val)} validation episodes possible, {n_val} asked")
    return sorted(val)


def episodes_of(repo_id, root):
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    meta = LeRobotDatasetMetadata(repo_id, root=root)
    return [(meta.episodes[ep]["tasks"][0], meta.episodes[ep]["length"]) for ep in range(meta.total_episodes)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.join(ROOT, "data", "lerobot", "gardener_v1"))
    ap.add_argument("--repo-id", default="Kimaa6/icub_gardener_v1")
    ap.add_argument("--val", type=int, default=20, help="validation episodes in total, kept ones included")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-val", default=None, help="earlier split JSON: its validation episodes stay validation")
    ap.add_argument("--old-root", default=None, help="the earlier split's dataset, to check its episodes match")
    ap.add_argument("--old-repo-id", default="Kimaa6/icub_gardener_v1")
    ap.add_argument("--out", default=os.path.join(ROOT, "configs", "act_split.json"))
    args = ap.parse_args()
    eps = episodes_of(args.repo_id, args.root)
    tasks = [t for t, _ in eps]
    keep, candidates = (), None
    if args.keep_val:
        old = json.load(open(args.keep_val))
        old_eps = sorted(old["train"] + old["val"])
        if args.old_root:
            old_info = episodes_of(args.old_repo_id, args.old_root)
            assert old_eps == list(range(len(old_info))), "the earlier split does not cover its dataset"
            wrong = [ep for ep in old_eps if ep >= len(eps) or eps[ep] != old_info[ep]]
            if wrong:
                raise SystemExit(f"episodes {wrong[:10]} differ from {args.old_root}: the old indices do not match")
            print(f"episodes 0-{len(old_info) - 1} match {args.old_root} (instruction and frame count)")
        keep = tuple(old["val"])
        candidates = [ep for ep in range(len(eps)) if ep not in set(old_eps)]
    val = choose_val(tasks, args.val, args.seed, keep, candidates)
    train = [ep for ep in range(len(eps)) if ep not in set(val)]
    assert not set(train) & set(val), "an episode is in both training and validation"
    assert sorted(train + val) == list(range(len(eps))), "an episode is missing"
    assert set(keep) <= set(val), "a kept validation episode is not in validation"
    with open(args.out, "w") as f:
        json.dump({"dataset": args.repo_id, "seed": args.seed, "train": train, "val": val}, f, indent=1)
    print(f"{len(train)} training and {len(val)} validation episodes ({len(keep)} kept, {len(val) - len(keep)} new), "
          f"none in both, saved to {args.out}")
    for t in sorted(set(tasks)):
        n_kept = sum(ep in keep for ep in val if tasks[ep] == t)
        n_val = sum(tasks[ep] == t for ep in val)
        print(f"  {t!r}: {n_val} val ({n_kept} kept, {n_val - n_kept} new), {sum(tasks[ep] == t for ep in train)} train")


if __name__ == "__main__":
    main()
