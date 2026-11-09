#!/bin/bash
# Train ACT on the gardener dataset with LeRobot, on the training episodes of configs/act_split.json.
#
# Same command on Kaggle, Lightning AI or the laptop; only the paths and sizes change. See docs/training.md.
#   DATASET=/path/to/gardener_v1 OUT=/path/to/outputs/act STEPS=80000 bash scripts/train_act.sh
#   SMOKE=1 bash scripts/train_act.sh       # laptop check: 200 steps, batch 4, nothing kept that matters
#
# Defaults follow the ACT paper and LeRobot (chunk of 100 actions = 4 s at 25 fps, lr 1e-5, KL weight 10),
# with mixed precision for a 16 GB T4. Checkpoints every SAVE steps: compare them with scripts/val_loss.py.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
DATASET="${DATASET:-$HERE/data/lerobot/gardener_v1}"
OUT="${OUT:-$HERE/outputs/act}"
STEPS="${STEPS:-80000}"
BATCH="${BATCH:-8}"
SAVE="${SAVE:-10000}"
WORKERS="${WORKERS:-4}"
DEVICE="${DEVICE:-cuda}"
AMP="${AMP:-true}"
if [ "${SMOKE:-0}" = 1 ]; then STEPS=200; BATCH=4; SAVE=100; OUT="${OUT}_smoke"; fi
SPLIT="${SPLIT:-$HERE/configs/act_split.json}"
TRAIN_EPISODES="$(python -c "import json;print(json.load(open('$SPLIT'))['train'])" | tr -d ' ')"
LEROBOT_TRAIN="${LEROBOT_TRAIN:-lerobot-train}"
echo "ACT: $STEPS steps, batch $BATCH, on $DEVICE, $(python -c "print(len($TRAIN_EPISODES))") training episodes, output $OUT"
"$LEROBOT_TRAIN" \
  --dataset.repo_id=Kimaa6/icub_gardener_v1 \
  --dataset.root="$DATASET" \
  --dataset.episodes="$TRAIN_EPISODES" \
  --policy.type=act \
  --policy.device="$DEVICE" \
  --policy.use_amp="$AMP" \
  --policy.push_to_hub=false \
  --output_dir="$OUT" \
  --job_name=act_gardener \
  --batch_size="$BATCH" \
  --steps="$STEPS" \
  --save_freq="$SAVE" \
  --log_freq=200 \
  --num_workers="$WORKERS" \
  --eval_freq=0 \
  --wandb.enable=false \
  --seed=1000
