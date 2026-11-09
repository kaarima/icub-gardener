#!/usr/bin/env python3
"""ACT's action error on the validation episodes, and on as many training episodes, for each checkpoint.

lerobot-train only reports the training loss. If the training error keeps falling while the validation
error rises, the policy is memorising its episodes; pick the checkpoint where validation is lowest. The error
is the mean absolute difference between predicted and recorded action chunks, in normalised units, with the
policy in evaluation mode (no VAE sampling), so it is lower than the training loss, which includes the KL term.

Run in the gardener-train env (a GPU helps but is not needed):
  env -u PYTHONPATH ~/miniconda3/envs/gardener-train/bin/python scripts/val_loss.py outputs/act/checkpoints/*/pretrained_model
"""
import argparse
import json
import os
import random

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--root", default=os.path.join(ROOT, "data", "lerobot", "gardener_v1"))
    ap.add_argument("--repo-id", default="Kimaa6/icub_gardener_v1")
    ap.add_argument("--split", default=os.path.join(ROOT, "configs", "act_split.json"))
    ap.add_argument("--batches", type=int, default=40, help="batches of 16 per set; random frames, fixed seed")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    split = json.load(open(args.split))
    meta = LeRobotDatasetMetadata(args.repo_id, root=args.root)
    train_check = random.Random(0).sample(split["train"], min(len(split["val"]), len(split["train"])))
    print(f"{len(split['val'])} validation episodes, compared with {len(train_check)} training episodes")
    for ckpt in args.checkpoints:
        config = PreTrainedConfig.from_pretrained(ckpt)
        config.device = args.device
        policy = get_policy_class(config.type).from_pretrained(ckpt, config=config).to(args.device).eval()
        pre, _ = make_pre_post_processors(policy.config, pretrained_path=ckpt,
                                          preprocessor_overrides={"device_processor": {"device": args.device}})
        delta = resolve_delta_timestamps(policy.config, meta)
        out = {}
        for name, episodes in (("val", split["val"]), ("train", train_check)):
            ds = LeRobotDataset(args.repo_id, root=args.root, episodes=episodes, delta_timestamps=delta)
            g = torch.Generator().manual_seed(0)
            loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=True, generator=g, num_workers=2)
            total, n = 0.0, 0
            with torch.no_grad():
                for k, batch in enumerate(loader):
                    if k >= args.batches:
                        break
                    b = pre(batch)
                    b["observation.images"] = [b[key] for key in policy.config.image_features]
                    # policy.forward would add the KL term, which needs the VAE that eval mode skips
                    predicted, _ = policy.model(b)
                    error = (predicted - b["action"]).abs() * ~b["action_is_pad"].unsqueeze(-1)
                    total += float(error.mean())
                    n += 1
            out[name] = total / max(n, 1)
        print(f"{ckpt}: validation {out['val']:.4f}, training {out['train']:.4f}, ratio {out['val'] / max(out['train'], 1e-9):.2f}", flush=True)


if __name__ == "__main__":
    main()
