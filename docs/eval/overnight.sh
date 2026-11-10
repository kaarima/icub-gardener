#!/bin/bash
# ACT + task ID overnight: train 80k steps (checkpoint every 10k), validation error at each checkpoint,
# then the 50 dev seeds for the 3 checkpoints with the lowest validation error. Held-out is NOT run here.
cd /home/karima/icub-gardener
[ "$(cat /sys/class/power_supply/AC/online)" = 1 ] || { echo "ON BATTERY: not starting"; exit 1; }
export PATH=~/miniconda3/envs/gardener-train/bin:$PATH
L=data/eval
# Charger watchdog: training on battery would drain it and die mid-write. Resume with lerobot-train --resume.
( while sleep 20; do [ "$(cat /sys/class/power_supply/AC/online)" = 1 ] || { echo "CHARGER UNPLUGGED at $(date): stopping"; pkill -f "[l]erobot-train"; pkill -f "[e]val_expert.py"; pkill -f "[r]un_expert.py"; exit; }; done ) &
WATCH=$!
echo "train start $(date)"
STEPS=80000 BATCH=8 SAVE=10000 LOG=500 OUT=outputs/act env -u PYTHONPATH bash scripts/train_act.sh > $L/act_train.log 2>&1
echo "train end $(date), exit $?"
grep -oE "step:[0-9KM.]+ .*loss:[0-9.]+" $L/act_train.log | sed 's/smpl.*loss/loss/' | awk 'NR%8==0'
ls outputs/act/checkpoints
echo "--- validation error per checkpoint $(date)"
env -u PYTHONPATH python scripts/val_loss.py outputs/act/checkpoints/0*/pretrained_model --batches 40 2>&1 | grep -E "validation|Error|Traceback" | tee $L/act_val.txt
TOP=$(python - <<'PY'
import re
rows=[(float(m[2]), m[1]) for m in (re.search(r"(\S+/pretrained_model): validation ([0-9.]+)", l) for l in open("data/eval/act_val.txt")) if m]
print(" ".join(p for _, p in sorted(rows)[:3]))
PY
)
echo "--- dev seeds for the 3 lowest validation errors: $TOP"
for C in $TOP; do
  S=$(basename $(dirname $C)); echo "dev $S start $(date)"
  ~/miniconda3/envs/gardener/bin/python scripts/eval_policy.py $C --first-seed 0 > $L/act_dev_$S.txt 2>&1
  grep -E "^POLICY|^SUCCESS|amount in the|failures by type" $L/act_dev_$S.txt
done
kill $WATCH
echo "OVERNIGHT-DONE $(date)"
