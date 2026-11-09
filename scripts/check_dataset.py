#!/usr/bin/env python3
"""Check a LeRobot dataset against the raw episodes it came from, and write a contact sheet to look at.

Checks, per episode: frame count, instruction text, state and action values against the raw file, and how
closely the decoded video matches the raw images (PSNR). Also checks fps, timestamps and normalisation stats.
With --sheet it saves frames from the start, middle and end of some episodes as one image.

Run in the gardener-train env:
  env -u PYTHONPATH ~/miniconda3/envs/gardener-train/bin/python scripts/check_dataset.py --sheet data/check.png
"""
import argparse
import glob
import os

import numpy as np
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def to_uint8(image):
    """CHW float in 0..1 to HWC uint8."""
    return (image.numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)


def psnr(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=os.path.join(ROOT, "data", "raw"))
    ap.add_argument("--root", default=os.path.join(ROOT, "data", "lerobot", "gardener_v1"))
    ap.add_argument("--repo-id", default="kaarima/icub_gardener_v1")
    ap.add_argument("--sheet", default=None, help="save a contact sheet PNG here")
    ap.add_argument("--images-per-episode", type=int, default=6, help="decoded frames compared per episode")
    args = ap.parse_args()

    ds = LeRobotDataset(args.repo_id, root=args.root)
    files = sorted(glob.glob(os.path.join(args.raw, "episode_*.npz")))
    print(f"loaded: {ds.num_episodes} episodes, {ds.num_frames} frames, {ds.fps} fps; features {list(ds.features)[:4]}")
    problems = []
    if ds.num_episodes != len(files):
        problems.append(f"{len(files)} raw files but {ds.num_episodes} episodes")

    index, rows, worst_psnr, worst_state = 0, [], 99.0, 0.0
    tasks = {}
    for ep, path in enumerate(files):
        e = np.load(path)
        n, text = len(e["state"]), str(e["instruction"])
        cameras = [c for c in ("eye", "wrist", "scene") if c in e]
        tasks[text] = tasks.get(text, 0) + 1
        start = index
        # Numbers, every frame, through the parquet tables only (no video decoding, so this is quick).
        table = ds.hf_dataset.select(range(start, start + n)).with_format("numpy")
        state = np.stack(table["observation.state"])
        action = np.stack(table["action"])
        worst_state = max(worst_state, float(np.abs(state - e["state"]).max()), float(np.abs(action - e["action"]).max()))
        times = np.asarray(table["timestamp"]).ravel()
        if not np.allclose(np.diff(times), 1.0 / ds.fps, atol=1e-3):
            problems.append(f"episode {ep}: uneven timestamps")
        for i in np.linspace(0, n - 1, args.images_per_episode).astype(int):
            item = ds[start + int(i)]
            if item["task"] != text:
                problems.append(f"episode {ep} frame {i}: task {item['task']!r}, expected {text!r}")
            for c in cameras:
                worst_psnr = min(worst_psnr, psnr(to_uint8(item[f"observation.images.{c}"]), e[c][i]))
        if len(rows) < 4:
            cells = []
            for f in (0, 0.3, 0.55, 0.7, 0.95):
                item = ds[start + int(f * (n - 1))]
                cells.append(np.vstack([to_uint8(item[f"observation.images.{c}"]) for c in cameras]))
            rows.append(np.hstack(cells))
        moves = float(np.abs(np.diff(e["action"][:, :10], axis=0)).max())
        print(f"episode {ep:2d} {os.path.basename(path)}: {n} frames ({n / ds.fps:.1f} s), largest joint target jump {np.degrees(moves):.2f} deg/frame, {text!r}")
        index += n
    if index != ds.num_frames:
        problems.append(f"raw files hold {index} frames, dataset {ds.num_frames}")
    if worst_state > 1e-5:
        problems.append(f"state or action differs from the raw file by {worst_state}")
    if worst_psnr < 30:
        problems.append(f"video quality low: PSNR {worst_psnr:.1f} dB")
    stats = ds.meta.stats
    print("state stats present:", "observation.state" in stats, "| action std (first 3):", np.round(np.asarray(stats["action"]["std"])[:3], 3))
    print(f"instructions: {tasks}")
    print(f"worst state/action difference from raw {worst_state:.2e}; worst decoded-image PSNR {worst_psnr:.1f} dB")
    if args.sheet:
        Image.fromarray(np.vstack(rows)).save(args.sheet)
        print(f"contact sheet: {args.sheet}")
    print("RESULT: " + ("all checks passed" if not problems else "PROBLEMS: " + "; ".join(problems)))


if __name__ == "__main__":
    main()
