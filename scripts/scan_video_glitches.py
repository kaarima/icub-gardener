#!/usr/bin/env python3
"""Read-only scan of dataset v2 for wrong-block glitches, every frame, every camera, episodes FIRST..LAST.

Per frame, with the image cut into 16x16-pixel blocks (14 x 14 blocks, mean absolute difference per block, 0-255):
  enc   = worst block of |decoded video - raw image|                       -> encoding error of any kind
  bias  = worst block of |mean signed (decoded - raw)|, averaged over colour -> a wrong flat colour (encoding glitch)
  spike = worst block of min(|raw[i]-raw[i-1]|, |raw[i]-raw[i+1]|) - |raw[i-1]-raw[i+1]|
          (one raw frame differs from both neighbours while they agree)   -> rendering glitch in the sim
Decoding uses torchcodec at mid-frame times (from_timestamp + (i + 0.5) / fps); checked against LeRobot's own decoding.
Writes per-frame values to OUT.npz; reads only. Next: scripts/glitch_regions.py sorts the candidates by the size of
the wrong region, and scripts/glitch_sheets.py shows them. Results for v2: docs/eval/v2_video_scan.txt.

  env -u PYTHONPATH ~/miniconda3/envs/gardener-train/bin/python scripts/scan_video_glitches.py 200 399 data/eval/v2_video_scan.npz
(about 18 min for 200 episodes)
"""
import glob
import sys
import time

import numpy as np
from torchcodec.decoders import VideoDecoder
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

ROOT, REPO, CAMS, B = "data/lerobot/gardener_v2", "Kimaa6/icub_gardener_v2", ("eye", "wrist", "scene"), 16


def blocks(diff):
    """Mean of |diff| over 16x16 blocks and colour: (n, 14, 14)."""
    n, h, w, _ = diff.shape
    return np.abs(diff).reshape(n, h // B, B, w // B, B, 3).mean(axis=(2, 4, 5))


def bias(diff):
    """Mean over colour of |mean signed difference| over 16x16 blocks: (n, 14, 14). High for a wrong flat colour."""
    n, h, w, _ = diff.shape
    return np.abs(diff.reshape(n, h // B, B, w // B, B, 3).mean(axis=(2, 4))).mean(axis=-1)


def main():
    first, last, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    meta = LeRobotDatasetMetadata(REPO, root=ROOT)
    files = sorted(glob.glob("data/raw/episode_*.npz"))
    assert len(files) == meta.total_episodes == 400
    decoders, res, t0 = {}, {}, time.time()
    for ep in range(first, last + 1):
        e = meta.episodes[ep]
        raw_file = np.load(files[ep])
        n = int(e["length"])
        for cam in CAMS:
            key = f"observation.images.{cam}"
            path = f"{ROOT}/" + meta.video_path.format(video_key=key, chunk_index=e[f"videos/{key}/chunk_index"],
                                                       file_index=e[f"videos/{key}/file_index"])
            if path not in decoders:
                decoders.clear()
                decoders[path] = VideoDecoder(path)
            raw = raw_file[cam]
            assert len(raw) == n, (ep, cam, len(raw), n)
            ts = (e[f"videos/{key}/from_timestamp"] + (np.arange(n) + 0.5) / meta.fps).tolist()  # mid-frame: from_timestamp has float error
            enc, bia = np.empty(n, np.float32), np.empty(n, np.float32)
            for s in range(0, n, 100):
                frames = decoders[path].get_frames_played_at(seconds=ts[s:s + 100]).data  # (k, 3, H, W) uint8
                dec = frames.permute(0, 2, 3, 1).numpy().astype(np.int16)
                diff = dec - raw[s:s + 100].astype(np.int16)
                enc[s:s + len(dec)] = blocks(diff).max(axis=(1, 2))
                bia[s:s + len(dec)] = bias(diff).max(axis=(1, 2))
            r = raw.astype(np.int16)
            prev, nxt, across = blocks(r[1:-1] - r[:-2]), blocks(r[1:-1] - r[2:]), blocks(r[2:] - r[:-2])
            spike = np.zeros(n, np.float32)
            spike[1:-1] = (np.minimum(prev, nxt) - across).max(axis=(1, 2))
            res[f"{ep}_{cam}_enc"], res[f"{ep}_{cam}_spike"], res[f"{ep}_{cam}_bias"] = enc, spike, bia
        res[f"{ep}_seed"] = np.array(int(files[ep][-9:-4]))
        print(f"episode {ep} (seed {files[ep][-9:-4]}): {n} frames; worst enc "
              + ", ".join(f"{c} {res[f'{ep}_{c}_enc'].max():.1f}" for c in CAMS)
              + "; worst bias " + ", ".join(f"{c} {res[f'{ep}_{c}_bias'].max():.1f}" for c in CAMS)
              + "; worst spike " + ", ".join(f"{c} {res[f'{ep}_{c}_spike'].max():.1f}" for c in CAMS)
              + f" | {time.time() - t0:.0f} s", flush=True)
    np.savez(out, **res)
    print("SCAN-DONE", out)


if __name__ == "__main__":
    main()
