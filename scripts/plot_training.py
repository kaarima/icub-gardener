"""Plot an ACT training run: training loss vs step, and validation vs training action error at each checkpoint.

Reads the lerobot-train log (loss every LOG steps) and the output of scripts/val_loss.py. Runs on the CPU.

Usage:
  python scripts/plot_training.py --log data/eval/act_train.log --val data/eval/act_val.txt \
      --out docs/media/act_training_curve.png
"""
import argparse
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE = "#2a78d6", "#eb6834"   # categorical slots 1 and 2
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5e0"


def steps_value(text):
    """'10K' -> 10000, '500' -> 500."""
    mult = {"K": 1e3, "M": 1e6}.get(text[-1], 1)
    return int(float(text.rstrip("KM")) * mult)


def read_loss(path):
    text = open(path, encoding="utf-8", errors="replace").read().replace("\r", "\n")
    pts = [(steps_value(s), float(l)) for s, l in re.findall(r"step:([0-9.]+[KM]?) .*?loss:([0-9.]+)", text)]
    return [p[0] for p in pts], [p[1] for p in pts]


def read_val(path):
    rows = re.findall(r"checkpoints/(\d+)/pretrained_model: validation ([0-9.]+), training ([0-9.]+)", open(path).read())
    return [int(s) for s, _, _ in rows], [float(v) for _, v, _ in rows], [float(t) for _, _, t in rows]


def style(ax, title):
    ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold")
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    ax.set_xlabel("training step", color=MUTED, fontsize=9)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", default="data/eval/act_train.log")
    ap.add_argument("--val", default="data/eval/act_val.txt")
    ap.add_argument("--out", default="docs/media/act_training_curve.png")
    ap.add_argument("--title", default="ACT + task ID, 80k steps, batch 8, 180 training episodes")
    args = ap.parse_args()

    steps, loss = read_loss(args.log)
    ck, val, trn = read_val(args.val)

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor="white")
    fig.suptitle(args.title, x=0.01, ha="left", color=INK, fontsize=12)

    a.plot(steps, loss, color=BLUE, linewidth=2)
    a.set_yscale("log")
    a.set_ylabel("loss (L1 + KL, log scale)", color=MUTED, fontsize=9)
    a.annotate(f"{loss[-1]:.3f}", (steps[-1], loss[-1]), xytext=(4, 6), textcoords="offset points", color=INK, fontsize=9)
    style(a, "Training loss")

    b.plot(ck, val, color=ORANGE, linewidth=2, marker="o", markersize=7, markeredgecolor="white", markeredgewidth=2,
           label="validation (20 held-back episodes)")
    b.plot(ck, trn, color=BLUE, linewidth=2, marker="o", markersize=7, markeredgecolor="white", markeredgewidth=2,
           label="training (20 matched episodes)")
    for ys, name in ((val, "validation"), (trn, "training")):
        b.annotate(f"{name}\n{ys[-1]:.3f}", (ck[-1], ys[-1]), xytext=(10, 0), textcoords="offset points",
                   ha="left", va="center", color=INK, fontsize=9)
    b.set_ylim(0, max(val) * 1.15)
    b.set_xlim(ck[0] - 4000, ck[-1] + 12000)
    b.set_ylabel("action error (normalised L1)", color=MUTED, fontsize=9)
    b.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower left")
    style(b, "Action error at each checkpoint")

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(args.out, dpi=150)
    print(f"{len(steps)} loss points, {len(ck)} checkpoints -> {args.out}")


if __name__ == "__main__":
    main()
