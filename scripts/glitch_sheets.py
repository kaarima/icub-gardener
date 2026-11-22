#!/usr/bin/env python3
"""Rank the scan's frames and save sheets of the worst ones: raw | decoded | difference x4, one row per frame.

For each candidate it also gives the 'shift' of the worst block: |mean of the signed difference| / mean |difference|.
A wrong-picture block (one flat colour replaced by another) has a shift near 1; ordinary compression error at
edges and water dots goes both ways and has a shift near 0.

  env -u PYTHONPATH ~/miniconda3/envs/gardener-train/bin/python scripts/glitch_sheets.py data/eval/v2_video_scan.npz bias 12 data/eval/sheet
KIND: enc, bias or spike. For spike the middle column is the average of the two neighbouring raw frames.
"""
import glob
import sys

import numpy as np
from PIL import Image
from torchcodec.decoders import VideoDecoder
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

ROOT, REPO, B = "data/lerobot/gardener_v2", "Kimaa6/icub_gardener_v2", 16

scan, kind, top, prefix = np.load(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4]
meta = LeRobotDatasetMetadata(REPO, root=ROOT)
files = sorted(glob.glob("data/raw/episode_*.npz"))
rows = []
for k in scan.files:
    if k.endswith("_" + kind):
        ep, cam = k.split("_")[:2]
        v = scan[k]
        rows += [(float(v[i]), int(ep), cam, int(i)) for i in np.argsort(v)[::-1][:5]]  # at most 5 per episode and camera
for cam in ("eye", "wrist", "scene"):
    allv = np.concatenate([scan[k] for k in scan.files if k.endswith(f"_{cam}_{kind}")])
    print(f"{cam}: {len(allv)} frames, {kind} median {np.median(allv):.1f}, 99.9th pct {np.percentile(allv, 99.9):.1f}, max {allv.max():.1f}")
rows.sort(reverse=True)
sheet = []
for value, ep, cam, i in rows[:top]:
    e = meta.episodes[ep]
    key = f"observation.images.{cam}"
    raw = np.load(files[ep])[cam]
    dec = VideoDecoder(f"{ROOT}/" + meta.video_path.format(video_key=key, chunk_index=e[f"videos/{key}/chunk_index"],
                                                         file_index=e[f"videos/{key}/file_index"]))
    d = dec.get_frames_played_at(seconds=[e[f"videos/{key}/from_timestamp"] + (i + 0.5) / meta.fps]).data[0].permute(1, 2, 0).numpy()
    if kind in ("enc", "bias"):
        a, b = raw[i], d
    else:                                   # spike: the raw frame against the average of its two neighbours
        a, b = raw[i], ((raw[i - 1].astype(int) + raw[i + 1].astype(int)) // 2).astype(np.uint8)
    diff = a.astype(int) - b.astype(int)
    blk = np.abs(diff).reshape(14, B, 14, B, 3).mean(axis=(1, 3, 4))
    by, bx = np.unravel_index(blk.argmax(), blk.shape)
    patch = diff[by * B:(by + 1) * B, bx * B:(bx + 1) * B]
    shift = abs(patch.mean(axis=(0, 1))).mean() / max(np.abs(patch).mean(), 1e-6)
    print(f"{kind} {value:5.1f}  episode {ep} seed {files[ep][-9:-4]} {cam:<5} frame {i:4d}/{len(raw)}  worst block ({by},{bx}) shift {shift:.2f}")
    sheet.append(np.hstack([a, b, np.clip(np.abs(diff) * 4, 0, 255).astype(np.uint8)]))
for s in range(0, len(sheet), 6):
    Image.fromarray(np.vstack(sheet[s:s + 6])).save(f"{prefix}_{s // 6}.png")
print("sheets:", (len(sheet) + 5) // 6)
