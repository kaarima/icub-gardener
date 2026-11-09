#!/usr/bin/env python3
"""Convert the raw episodes from record_episodes.py into a LeRobot dataset.

Run in the gardener-train env, with your ROS PYTHONPATH cleared so its old packages do not get imported:
  env -u PYTHONPATH ~/miniconda3/envs/gardener-train/bin/python scripts/to_lerobot.py

Each frame also holds the instruction as five numbers (observation.environment_state), for ACT, which cannot
read the text; see gardener/data/instructions.py. The text itself is the frame's task, for SmolVLA.
Frames go straight into the video encoder, with no PNG files in between (--png uses the old way, to compare).
Nothing is uploaded. The dataset is written to data/lerobot/gardener_v1, replacing any older copy there.
"""
import argparse
import glob
import os
import shutil

import sys

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gardener.data.instructions import CODE_NAMES, code  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=os.path.join(ROOT, "data", "raw"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "lerobot", "gardener_v1"))
    ap.add_argument("--repo-id", default="kaarima/icub_gardener_v1", help="dataset name; not uploaded by this script")
    ap.add_argument("--png", action="store_true", help="write each frame as a PNG first, as LeRobot does by default")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.raw, "episode_*.npz")))
    if not files:
        raise SystemExit(f"no episodes in {args.raw}")
    first = np.load(files[0])
    names = [str(n) for n in first["joints"]]
    h, w, _ = first["eye"].shape[1:]
    cameras = [c for c in ("eye", "wrist", "scene") if c in first]
    features = {
        **{f"observation.images.{c}": {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channels"]} for c in cameras},
        "observation.state": {"dtype": "float32", "shape": (len(names),), "names": names},
        "observation.environment_state": {"dtype": "float32", "shape": (len(CODE_NAMES),), "names": CODE_NAMES},
        "action": {"dtype": "float32", "shape": (len(names),), "names": names},
    }
    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    ds = LeRobotDataset.create(repo_id=args.repo_id, fps=int(first["fps"]), features=features, root=args.out,
                               robot_type="icub_upper_body", use_videos=True, streaming_encoding=not args.png)
    for path in files:
        e = np.load(path)
        task = str(e["instruction"])
        # Read each camera once: e[c] unpacks the whole array from the file every time it is used.
        frames = {c: e[c] for c in cameras}
        state, action = e["state"], e["action"]
        instruction_code = np.asarray(code(task), dtype=np.float32)
        for i in range(len(state)):
            ds.add_frame({**{f"observation.images.{c}": frames[c][i] for c in cameras},
                          "observation.state": state[i], "observation.environment_state": instruction_code,
                          "action": action[i], "task": task})
        ds.save_episode()
        print(f"{os.path.basename(path)}: {len(state)} frames, {task!r}", flush=True)
    ds.finalize()
    print(f"dataset in {args.out}: {ds.meta.total_episodes} episodes, {ds.meta.total_frames} frames")


if __name__ == "__main__":
    main()
