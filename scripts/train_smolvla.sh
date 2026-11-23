#!/bin/bash
# Fine-tune SmolVLA (lerobot/smolvla_base) on the gardener dataset with LeRobot, on the training episodes of a split.
#
# Same data as ACT v1: dataset v1, the 180 training episodes of configs/act_split.json, training seed 1000.
# What SmolVLA gets: the 3 cameras, the 12 joint values, and the instruction as a sentence ("Water the left plant.").
# It does NOT get ACT's task-ID vector (observation.environment_state): the pretrained model has no input for it, so
# the target pot and the amount must come from the words.
#
# The base model was trained with cameras named camera1-3, so ours are renamed to those: eye -> camera1,
# wrist -> camera2, scene -> camera3. The rename is saved in the checkpoint's preprocessor, so at evaluation the
# simulation keeps sending eye, wrist and scene.
#
# Settings are LeRobot's SmolVLA defaults (frozen vision encoder, only the action expert trained, lr 1e-4 with
# cosine decay, chunk of 50 actions = 2 s at 25 fps), except the number of steps and the batch, which are set
# to fit a Kaggle T4 (see notebooks/train_smolvla_v1_kaggle.ipynb).
#   DATASET=/path/to/gardener_v1 OUT=/path/to/outputs/smolvla STEPS=20000 BATCH=32 bash scripts/train_smolvla.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
DATASET="${DATASET:-$HERE/data/lerobot/gardener_v1}"
OUT="${OUT:-$HERE/outputs/smolvla}"
STEPS="${STEPS:-20000}"
BATCH="${BATCH:-32}"
SAVE="${SAVE:-4000}"
WORKERS="${WORKERS:-4}"
DEVICE="${DEVICE:-cuda}"
SPLIT="${SPLIT:-$HERE/configs/act_split.json}"
REPO_ID="${REPO_ID:-Kimaa6/icub_gardener_v1}"
TRAIN_EPISODES="$(python -c "import json;print(json.load(open('$SPLIT'))['train'])" | tr -d ' ')"
RENAME='{"observation.images.eye": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2", "observation.images.scene": "observation.images.camera3"}'
echo "SmolVLA: $STEPS steps, batch $BATCH, on $DEVICE, $(python -c "print(len($TRAIN_EPISODES))") training episodes, output $OUT"
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.device="$DEVICE" \
  --policy.push_to_hub=false \
  --dataset.repo_id="$REPO_ID" \
  --dataset.root="$DATASET" \
  --dataset.episodes="$TRAIN_EPISODES" \
  --rename_map="$RENAME" \
  --output_dir="$OUT" \
  --job_name=smolvla_gardener \
  --batch_size="$BATCH" \
  --steps="$STEPS" \
  --save_freq="$SAVE" \
  --log_freq="${LOG:-100}" \
  --num_workers="$WORKERS" \
  --eval_freq=0 \
  --wandb.enable=false \
  --seed=1000
