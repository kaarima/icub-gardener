#!/usr/bin/env python3
"""For candidate frames (bias over 6), count 16x16 blocks whose colour is shifted (|mean signed diff| > 4), and the
largest connected group of them. A wrong-picture region covers many neighbouring blocks; water-dot blur only a few.
Also checks the raw frame against its neighbours in the same region: if the raw image agrees with them there, the
fault is in the video (encoding), not in the render. Read-only.

  env -u PYTHONPATH ~/miniconda3/envs/gardener-train/bin/python scripts/glitch_regions.py data/eval/v2_video_scan.npz docs/eval/v2_video_scan_regions.json

On v2, water-dot blur gives 1-5 neighbouring blocks; the wrong-picture glitches gave 14-24."""
import glob, json, sys
import numpy as np
from torchcodec.decoders import VideoDecoder
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
def label(mask):
    """Connected groups (4-neighbour) of True cells in a small 2-D grid: (labels, count), like scipy.ndimage.label."""
    lab, n = np.zeros(mask.shape, int), 0
    for y, x in zip(*np.nonzero(mask)):
        if lab[y, x]:
            continue
        n += 1; lab[y, x] = n; stack = [(y, x)]
        while stack:
            cy, cx = stack.pop()
            for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
                if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1] and mask[ny, nx] and not lab[ny, nx]:
                    lab[ny, nx] = n; stack.append((ny, nx))
    return lab, n


ROOT, REPO, B, _ = "data/lerobot/gardener_v2", "Kimaa6/icub_gardener_v2", 16, None
scan = np.load(sys.argv[1]); meta = LeRobotDatasetMetadata(REPO, root=ROOT); files = sorted(glob.glob("data/raw/episode_*.npz"))
cands = sorted((int(k.split("_")[0]), k.split("_")[1], int(i), float(scan[k][i])) for k in scan.files if k.endswith("_bias") for i in np.where(scan[k] > 6)[0])
print(len(cands), "candidate frames")
out, raw_cache = [], {}
for ep, cam, i, b in cands:
    e = meta.episodes[ep]; key = f"observation.images.{cam}"
    if (ep, cam) not in raw_cache:
        raw_cache.clear(); raw_cache[(ep, cam)] = np.load(files[ep])[cam]
    raw = raw_cache[(ep, cam)].astype(int)
    dec = VideoDecoder(f"{ROOT}/" + meta.video_path.format(video_key=key, chunk_index=e[f"videos/{key}/chunk_index"], file_index=e[f"videos/{key}/file_index"]))
    d = dec.get_frames_played_at(seconds=[e[f"videos/{key}/from_timestamp"] + (i + 0.5) / meta.fps]).data[0].permute(1, 2, 0).numpy().astype(int)
    shifted = np.abs((d - raw[i]).reshape(14, B, 14, B, 3).mean(axis=(1, 3))).mean(-1) > 4
    lab, n = label(shifted)
    largest = int(max((lab == k).sum() for k in range(1, n + 1))) if n else 0
    # raw frame vs the average of its neighbours, over the largest shifted region (is the raw image itself odd there?)
    region = np.kron(lab == (np.bincount(lab.ravel())[1:].argmax() + 1), np.ones((B, B), bool)) if n else np.zeros((224, 224), bool)
    nb = (raw[max(i - 1, 0)] + raw[min(i + 1, len(raw) - 1)]) / 2
    raw_vs_nb = float(np.abs(raw[i] - nb)[region].mean()) if n else 0.0
    dec_vs_raw = float(np.abs(d - raw[i])[region].mean()) if n else 0.0
    out.append(dict(ep=ep, seed=int(files[ep][-9:-4]), cam=cam, frame=i, bias=round(b, 1), shifted=int(shifted.sum()), largest=largest,
                    region_dec_vs_raw=round(dec_vs_raw, 1), region_raw_vs_neighbours=round(raw_vs_nb, 1)))
json.dump(out, open(sys.argv[2], "w"), indent=0)
for r in sorted(out, key=lambda r: -r["largest"]):
    if r["largest"] >= 6:
        print(r)
