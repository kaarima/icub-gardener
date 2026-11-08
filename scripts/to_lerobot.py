#!/usr/bin/env python3
"""Convert the raw episodes from record_episodes.py into a LeRobot dataset.

Run in the gardener-train env, with your ROS PYTHONPATH cleared so its old packages do not get imported:
  env -u PYTHONPATH ~/miniconda3/envs/gardener-train/bin/python scripts/to_lerobot.py

Nothing is uploaded. The dataset is written to data/lerobot/gardener_v1, replacing any older copy there.
"""
import argparse
import glob
import os
import shutil

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=os.path.join(ROOT, "data", "raw"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "lerobot", "gardener_v1"))
    ap.add_argument("--repo-id", default="kaarima/icub_gardener_v1", help="dataset name; not uploaded by this script")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.raw, "episode_*.npz")))
    if not files:
        raise SystemExit(f"no episodes in {args.raw}")
    first = np.load(files[0])
    names = [str(n) for n in first["joints"]]
    h, w, _ = first["eye"].shape[1:]
    features = {
        "observation.images.eye": {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channels"]},
        "observation.images.scene": {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channels"]},
        "observation.state": {"dtype": "float32", "shape": (len(names),), "names": names},
        "action": {"dtype": "float32", "shape": (len(names),), "names": names},
    }
    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    ds = LeRobotDataset.create(repo_id=args.repo_id, fps=int(first["fps"]), features=features, root=args.out,
                               robot_type="icub_upper_body", use_videos=True)
    for path in files:
        e = np.load(path)
        task = str(e["instruction"])
        for i in range(len(e["state"])):
            ds.add_frame({"observation.images.eye": e["eye"][i], "observation.images.scene": e["scene"][i],
                          "observation.state": e["state"][i], "action": e["action"][i], "task": task})
        ds.save_episode()
        print(f"{os.path.basename(path)}: {len(e['state'])} frames, {task!r}", flush=True)
    ds.finalize()
    print(f"dataset in {args.out}: {ds.meta.total_episodes} episodes, {ds.meta.total_frames} frames")


if __name__ == "__main__":
    main()
