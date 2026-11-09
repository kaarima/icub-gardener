#!/usr/bin/env python3
"""Make an image sheet from one recorded episode: eye camera above, scene camera below, phase on each frame.

Reads a raw episode from record_episodes.py and changes nothing. Phases come from the finger and thumb
commands saved in the episode, plus the expert's fixed timings, so the pour's start and end are within a
few frames. Needs the raw .npz file, so run it before deleting raw episodes.

Usage:
  python scripts/episode_sheet.py data/raw/episode_01000.npz --out data/episode_sheet.png
"""
import argparse
import os

import numpy as np
from PIL import Image, ImageDraw

FPS = 25
LABEL_HEIGHT = 44


def phase_ranges(state: np.ndarray) -> list:
    """[(phase name, first frame, last frame)] from the saved finger and thumb commands."""
    fingers, thumb = state[:, 10], state[:, 11]
    n = len(state)
    close_start = int(np.argmax(fingers > 0))
    thumb_full = int(np.argmax(thumb >= thumb.max()))
    release_start = n - int(np.argmax(thumb[::-1] >= thumb.max()))
    retract_start = n - int(np.argmax(fingers[::-1] >= 1.0))
    reach_end = 2 * FPS
    lift_start = thumb_full + 1 * FPS            # 1 s holding the closed grip
    transport_start = lift_start + 2 * FPS       # 1.5 s lift, 0.5 s check
    pour_start = transport_start + int(8.3 * FPS)     # 5.6 s carrying, 0.5 s settling, 2 corrections of 1.1 s
    return_start = release_start - int(7.9 * FPS)     # 5.9 s retracing the path, 2 s lowering
    return [("reach", 0, reach_end), ("approach", reach_end, close_start), ("grasp: fingers", close_start, thumb_full - 31),
            ("grasp: thumb last", thumb_full - 31, lift_start), ("lift check", lift_start, transport_start),
            ("transport", transport_start, pour_start), ("pour", pour_start, return_start), ("return", return_start, release_start),
            ("release", release_start, retract_start + int(1.5 * FPS)), ("retract", retract_start + int(1.5 * FPS), n)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episode")
    ap.add_argument("--out", default="data/episode_sheet.png")
    args = ap.parse_args()

    e = np.load(args.episode)
    state = e["state"]
    ranges = phase_ranges(state)
    wanted = ["reach", "approach", "grasp: fingers", "grasp: thumb last", "lift check", "transport", "pour", "return"]
    picks = []
    for name, a, b in ranges:
        if name in wanted:
            picks.append((name, a, (a + b) // 2))

    size = e["eye"].shape[1]
    title_height = 30
    cameras = [c for c in ("eye", "wrist", "scene") if c in e]
    sheet = Image.new("RGB", (size * len(picks), title_height + LABEL_HEIGHT + len(cameras) * size), "white")
    draw = ImageDraw.Draw(sheet)
    total = len(state) / FPS
    draw.text((6, 8), f"{os.path.basename(args.episode)}: {str(e['instruction'])!r}, {total:.1f} s. "
                      f"Rows, top to bottom: {', '.join(cameras)} camera.", fill="black")
    for k, (name, _, i) in enumerate(picks):
        x = k * size
        draw.text((x + 6, title_height + 4), f"{k + 1}. {name}", fill="black")
        draw.text((x + 6, title_height + 22), f"t = {i / FPS:.1f} s, frame {i}", fill=(90, 90, 90))
        for row, c in enumerate(cameras):
            sheet.paste(Image.fromarray(e[c][i]), (x, title_height + LABEL_HEIGHT + row * size))
    sheet.save(args.out)
    print("phases (seconds): " + ", ".join(f"{n} {a / FPS:.1f}-{b / FPS:.1f}" for n, a, b in ranges))
    print(f"saved {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
