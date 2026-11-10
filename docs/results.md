# Results

## ACT + task ID (first policy), in progress, 2026-11-10

**Training:**
- ACT, 80k steps, batch 8, on the laptop (RTX 3050, 4 GB), about 4 h 20 min.
- 180 training episodes and 20 validation episodes (`configs/act_split.json`).
- The instruction goes in as five numbers (task ID).
- Curve: `docs/media/act_training_curve.png`.

**Choosing the checkpoint:**
- Validation error was measured at every checkpoint (`docs/eval/act_val.txt`).
- The 3 checkpoints with the lowest validation error were run on the 50 dev seeds (0-49).
- Rule: highest dev success wins, and a tie goes to the lower validation error.
- Held-out seeds 5000-5049 are run **once**, on the chosen checkpoint only.

| Checkpoint | Validation error | Dev success | Amount in band | Dev failures |
|---|---|---|---|---|
| 80k | 0.0986 | 39/50 = 78% (65-87%) | 23/39 | 8 water missed, 2 can not upright, 1 collision |
| 60k | 0.1033 | 32/50 = 64% (50-76%) | 11/32 | 14 water missed, 4 collisions |
| 70k | 0.1040 | 31/50 = 62% (48-74%) | 14/31 | 16 water missed, 3 collisions |

**Picked: 80k.** It has the highest dev success and also the lowest validation error (`docs/eval/act_pick.txt`).
Held-out run started 2026-11-10 04:04.

On dev, 80k never chose the wrong pot (0/50). Its weak point is "just a little": only 3 of 16 of those pours landed in band.

The held-out table against the expert comes next. The expert's held-out report is in `docs/eval/expert_heldout.txt`.

**How to resume if the PC was switched off:**
- Checkpoints are in `outputs/act/checkpoints/` (on disk, not in git).
- Finish any missing dev run:
  `python scripts/eval_policy.py outputs/act/checkpoints/070000/pretrained_model --first-seed 0 > data/eval/act_dev_070000.txt`
- Then start `bash data/eval/heldout.sh` (copy in `docs/eval/heldout.sh`). It picks the checkpoint by the rule above and runs held-out once.
