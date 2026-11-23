#!/usr/bin/env python3
"""Offline action error of ACT checkpoints on the validation demos, in degrees, split by can yaw and by phase.

Why: ACT trained on dataset v2 fails at negative can yaw in simulation, while ACT v1 does not. This checks, without
any simulation, whether v2 already predicts the expert's actions worse there (it has not learned those demos well),
or just as well (the trouble only appears when the policy drives the robot itself and small errors add up).

For every 10th frame of each validation episode, each policy predicts its 100-step action chunk from the recorded
images and state. The error is |predicted - recorded| for the 10 arm and torso joints, in degrees, averaged over
the chunk (padding past the episode end left out). Each policy is un-normalised with its own action statistics.
Phase "pour" = chunks that start between 2 s before the pour and its end, from data/eval/pour_demos.npz
(scripts/compare_pour_demos.py); everything else is "other".
All 40 validation episodes were unseen by both policies (v1's 20 validation episodes are kept in v2's validation).

Run in the gardener-train env:
  env -u PYTHONPATH ~/miniconda3/envs/gardener-train/bin/python scripts/action_error_by_yaw.py \
      outputs/act/checkpoints/080000/pretrained_model outputs/act_v2/checkpoints/060000/pretrained_model
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BINS = [(-25, -10), (-10, 0), (0, 10), (10, 25)]
ARM = 10  # torso (3) + right arm (7); fingers and thumb are open/closed and left out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--root", default=os.path.join(ROOT, "data", "lerobot", "gardener_v2"))
    ap.add_argument("--repo-id", default="Kimaa6/icub_gardener_v2")
    ap.add_argument("--split", default=os.path.join(ROOT, "configs", "act_split_v2.json"))
    ap.add_argument("--raw", default=os.path.join(ROOT, "data", "raw"))
    ap.add_argument("--pours", default=os.path.join(ROOT, "data", "eval", "pour_demos.npz"))
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "eval", "action_error_by_yaw.json"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from safetensors.torch import load_file

    seeds = [int(os.path.basename(p)[8:13]) for p in sorted(glob.glob(os.path.join(args.raw, "episode_*.npz")))]
    pours = np.load(args.pours)
    pour_of = {int(s): (float(a), float(a + b)) for s, a, b in zip(pours["seed"], pours["pour_start_s"], pours["pour_s"])}
    val = json.load(open(args.split))["val"]
    meta = LeRobotDatasetMetadata(args.repo_id, root=args.root)

    policies = []
    for ckpt in args.checkpoints:
        config = PreTrainedConfig.from_pretrained(ckpt)
        config.device = args.device
        policy = get_policy_class(config.type).from_pretrained(ckpt, config=config).to(args.device).eval()
        pre, _ = make_pre_post_processors(policy.config, pretrained_path=ckpt,
                                          preprocessor_overrides={"device_processor": {"device": args.device}})
        stats = load_file(os.path.join(ckpt, "policy_postprocessor_step_0_unnormalizer_processor.safetensors"))
        policies.append((ckpt, policy, pre, stats["action.mean"].to(args.device), stats["action.std"].to(args.device)))
    delta = resolve_delta_timestamps(policies[0][1].config, meta)

    ds = LeRobotDataset(args.repo_id, root=args.root, episodes=val, delta_timestamps=delta)
    ep = np.array(ds.hf_dataset["episode_index"])
    fr = np.array(ds.hf_dataset["frame_index"])
    keep = np.flatnonzero(fr % args.every == 0)
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, keep), batch_size=16, num_workers=4)
    fps = meta.fps
    rows = []  # (episode, frame, error per policy)
    with torch.no_grad():
        for k, batch in enumerate(loader):
            errs = []
            for _, policy, pre, mean, std in policies:
                b = pre(dict(batch))
                b["observation.images"] = [b[key] for key in policy.config.image_features]
                predicted, _ = policy.model(b)
                predicted = predicted * std + mean
                truth = batch["action"].to(args.device)
                ok = (~b["action_is_pad"]).unsqueeze(-1).float()
                e = (predicted - truth).abs()[..., :ARM] * ok
                errs.append((e.sum((1, 2)) / (ok.sum((1, 2)) * ARM)).rad2deg().cpu().numpy())
            for i in range(len(batch["episode_index"])):
                rows.append((int(batch["episode_index"][i]), int(batch["frame_index"][i]), [float(x[i]) for x in errs]))
            if k % 20 == 0:
                print(f"{len(rows)}/{len(keep)} frames", flush=True)

    names = [os.path.relpath(c, ROOT) for c in args.checkpoints]
    table = []
    for e, f, errs in rows:
        seed = seeds[e]
        start, end = pour_of[seed]
        t = f / fps
        table.append(dict(episode=e, seed=seed, yaw=float(np.random.default_rng(seed).uniform(-25, 25)), frame=f,
                          phase="pour" if start - 2.0 <= t <= end else "other", err=errs))
    json.dump(dict(checkpoints=names, rows=table), open(args.out, "w"))

    def mean_err(sel, j):
        return np.mean([r["err"][j] for r in sel]) if sel else float("nan")

    print(f"\nmean arm action error in degrees over the 100-step chunk; {len(set(r['episode'] for r in table))} validation episodes")
    print("columns: " + " | ".join(names))
    for group, test in (("v1 validation episodes (0-199)", lambda r: r["seed"] < 1204),
                        ("new validation episodes (200-399)", lambda r: r["seed"] >= 1204), ("all", lambda r: True)):
        print(f"\n{group}")
        for a, b in BINS:
            for phase in ("pour", "other"):
                sel = [r for r in table if test(r) and a <= r["yaw"] < b and r["phase"] == phase]
                n_ep = len(set(r["episode"] for r in sel))
                print(f"  yaw {a:>3}..{b:>3} {phase:5} episodes {n_ep:2}: " +
                      " | ".join(f"{mean_err(sel, j):5.2f}" for j in range(len(names))))
    print(f"\nper-frame table: {args.out}")


if __name__ == "__main__":
    main()
