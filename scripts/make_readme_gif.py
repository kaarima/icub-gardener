#!/usr/bin/env python3
"""Make the README animation from one recorded expert episode: scene, eye and wrist cameras side by side.

The frames are the recorded ones (data/raw/episode_<seed>.npz, 224 x 224 per camera, 25 fps), so the GIF shows
exactly what the policies see. It is sped up (--speed) and thinned to --fps, and the instruction is written on top.
GIF colours come from one palette made over the whole clip (ffmpeg palettegen), which keeps the file small.
It also writes an .mp4 beside it, in case the GIF is too big.
For the project website: --scale 2 --mp4-only makes a larger H.264 video only, plus a poster image (.jpg).

Run in the gardener env (needs ffmpeg on the PATH):
  python scripts/make_readme_gif.py --seed 1000 --out docs/media/expert_episode.gif
  python scripts/make_readme_gif.py --seed 1000 --speed 2 --fps 25 --scale 2 --mp4-only --out docs/site/demo.gif
"""
import argparse
import os
import subprocess

import numpy as np

CAMERAS = ("scene", "eye", "wrist")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--speed", type=float, default=3.0, help="times real time")
    ap.add_argument("--fps", type=int, default=12, help="frames per second of the GIF")
    ap.add_argument("--out", default="docs/media/expert_episode.gif")
    ap.add_argument("--scale", type=int, default=1, help="enlarge the frames this many times (Lanczos)")
    ap.add_argument("--mp4-only", action="store_true", help="write the .mp4 and a poster .jpg, no GIF")
    args = ap.parse_args()
    z = np.load(os.path.join(args.raw, f"episode_{args.seed:05d}.npz"), allow_pickle=True)
    fps = float(z["fps"])
    step = max(1, round(fps * args.speed / args.fps))
    frames = np.concatenate([z[c] for c in CAMERAS], axis=2)[::step]   # (n, 224, 672, 3), side by side
    n, h, w, _ = frames.shape
    text = str(z["instruction"]).replace(":", r"\:").replace("'", "")
    label = (f"scale=iw*{args.scale}:ih*{args.scale}:flags=lanczos,"
             f"drawtext=text='{text}  ({args.speed:g}x)':x=8:y=8:fontsize={16 * args.scale}:fontcolor=white:"
             f"box=1:boxcolor=black@0.5:boxborderw={4 * args.scale}")
    raw_in = ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(args.fps), "-i", "-"]
    gif = ["ffmpeg", "-y", "-loglevel", "error", *raw_in, "-vf",
           f"{label},split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=4",
           "-loop", "0", args.out]
    mp4 = ["ffmpeg", "-y", "-loglevel", "error", *raw_in, "-vf", label, "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-crf", "23", "-movflags", "+faststart", os.path.splitext(args.out)[0] + ".mp4"]
    poster = ["ffmpeg", "-y", "-loglevel", "error", *raw_in, "-vf", f"select=eq(n\\,{2 * n // 3}),{label}",
              "-frames:v", "1", "-q:v", "3", os.path.splitext(args.out)[0] + ".jpg"]
    cmds = (mp4, poster) if args.mp4_only else (gif, mp4)
    for cmd in cmds:
        subprocess.run(cmd, input=frames.tobytes(), check=True)
    for path in [c[-1] for c in cmds]:
        print(f"{path}: {n} frames, {n / args.fps:.1f} s, {w * args.scale}x{h * args.scale}, {os.path.getsize(path) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
