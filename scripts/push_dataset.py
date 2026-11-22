#!/usr/bin/env python3
"""Upload the dataset to the Hugging Face Hub as a PRIVATE dataset repo. You run this yourself, once.

First log in with a token that can write (https://huggingface.co/settings/tokens), in the gardener-train env:
  env -u PYTHONPATH ~/miniconda3/envs/gardener-train/bin/hf auth login
Then:
  env -u PYTHONPATH ~/miniconda3/envs/gardener-train/bin/python scripts/push_dataset.py
  # dataset v2, with its card text (shown on the Hub page)
  env -u PYTHONPATH ~/miniconda3/envs/gardener-train/bin/python scripts/push_dataset.py --root data/lerobot/gardener_v2 \\
      --repo-id Kimaa6/icub_gardener_v2 --description docs/datasets/gardener_v2_card.md
On a slow or flaky connection add --large: it uploads file by file and a rerun skips the files already done
(a plain upload that stalls starts again from zero). With --large, --workers sets how many files go up at once:
keep it small on a slow line, or each connection gets so little speed that the storage server drops it.
It prints the size first and asks before uploading. Check afterwards on huggingface.co that the repo
shows "Private".
"""
import argparse
import os
import subprocess

from lerobot.datasets.lerobot_dataset import LeRobotDataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.join(ROOT, "data", "lerobot", "gardener_v1"))
    ap.add_argument("--repo-id", default="Kimaa6/icub_gardener_v1")
    ap.add_argument("--description", default=None, help="Markdown file shown as the description on the dataset card")
    ap.add_argument("--large", action="store_true", help="resumable upload, file by file (for slow connections)")
    ap.add_argument("--workers", type=int, default=None, help="with --large: files uploaded at once (default: CPU count - 2)")
    ap.add_argument("--yes", action="store_true", help="do not ask before uploading")
    args = ap.parse_args()
    ds = LeRobotDataset(args.repo_id, root=args.root)
    size = subprocess.run(["du", "-sh", args.root], capture_output=True, text=True).stdout.split()[0]
    print(f"{args.repo_id}: {ds.num_episodes} episodes, {ds.num_frames} frames, {size}, from {args.root}")
    print("It will be uploaded as a PRIVATE dataset repo.")
    if not args.yes and input("Upload? [y/N] ").strip().lower() != "y":
        raise SystemExit("not uploaded")
    if args.workers:  # LeRobot does not pass num_workers on, so set it on the Hub client
        import functools
        from huggingface_hub import HfApi
        HfApi.upload_large_folder = functools.partialmethod(HfApi.upload_large_folder, num_workers=args.workers)
    card = {"dataset_description": open(args.description).read()} if args.description else {}
    ds.push_to_hub(private=True, tags=["lerobot", "icub", "simulation"], license="apache-2.0",
                   upload_large_folder=args.large, **card)
    print(f"done: https://huggingface.co/datasets/{args.repo_id} (check that it says Private)")


if __name__ == "__main__":
    main()
