#!/bin/bash
# Evaluation settings for ACT + task ID (80k checkpoint), one at a time on the 50 dev seeds:
# n_action_steps 25, n_action_steps 50, temporal ensembling 0.01. Then the pick: highest dev success wins
# against the default setting's 39/50 (data/eval/act_dev_080000.txt); a tie goes to the default.
# Held-out is NOT run here: it is run once, by hand, on the pick.
cd /home/karima/icub-gardener
[ "$(cat /sys/class/power_supply/AC/online)" = 1 ] || { echo "ON BATTERY: not starting"; exit 1; }
PY=~/miniconda3/envs/gardener/bin/python
CK=outputs/act/checkpoints/080000/pretrained_model
( while sleep 20; do [ "$(cat /sys/class/power_supply/AC/online)" = 1 ] || { echo "CHARGER UNPLUGGED at $(date): stopping"; pkill -f "[e]val_expert.py"; pkill -f "[r]un_expert.py"; exit; }; done ) &
WATCH=$!
for S in "n25 --n-action-steps 25" "n50 --n-action-steps 50" "ens --ensemble 0.01"; do
  set -- $S; NAME=$1; shift
  echo "dev $NAME ($*) start $(date)"
  $PY scripts/eval_policy.py $CK --first-seed 0 "$@" > data/eval/act_dev_080000_$NAME.txt 2>&1
  grep -E "^POLICY|^SUCCESS|amount in the|failures by type|episodes with each|^speed" data/eval/act_dev_080000_$NAME.txt
done
kill $WATCH
$PY - <<'PY' | tee data/eval/act_settings_pick.txt
import re
rows = []
for name, f in [("default (100)", "act_dev_080000.txt"), ("n_action_steps 25", "act_dev_080000_n25.txt"),
                ("n_action_steps 50", "act_dev_080000_n50.txt"), ("ensemble 0.01", "act_dev_080000_ens.txt")]:
    t = open("data/eval/" + f).read()
    m = re.search(r"SUCCESS: (\d+)/(\d+)", t); b = re.search(r"amount in the instruction's band: (\d+)/(\d+)", t)
    rows.append((int(m[1]), name, b[1] + "/" + b[2] if b else "?"))
for ok, name, band in rows:
    print(f"{name:<20} dev {ok}/50, amount in band {band}")
best = max(rows, key=lambda r: (r[0], r[1].startswith("default")))
print("PICK", best[1], "(rule: highest dev success; a tie goes to the default)")
PY
echo "SETTINGS-DONE $(date)"
