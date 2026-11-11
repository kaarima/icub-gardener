#!/bin/bash
# After the overnight dev runs: pick the checkpoint (highest dev success; tie -> lower validation error),
# then run the 50 held-out seeds ONCE. Writes data/eval/act_heldout.txt and act_pick.txt.
cd /home/karima/icub-gardener
while pgrep -f "[o]vernight.sh" >/dev/null; do sleep 30; done
[ "$(cat /sys/class/power_supply/AC/online)" = 1 ] || { echo "ON BATTERY: not starting held-out"; exit 1; }
~/miniconda3/envs/gardener/bin/python - <<'PY' | tee data/eval/act_pick.txt
import re, glob
val = {m[1]: float(m[2]) for m in (re.search(r"checkpoints/(\d+)/pretrained_model: validation ([0-9.]+)", l) for l in open("data/eval/act_val.txt")) if m}
rows = []
for f in sorted(glob.glob("data/eval/act_dev_*.txt")):
    s = re.search(r"act_dev_(\d+)", f)[1]
    m = re.search(r"SUCCESS: (\d+)/(\d+)", open(f).read())
    if m: rows.append((int(m[1]), -val[s], s))
for ok, v, s in sorted(rows, reverse=True): print(f"{s}: dev {ok}/50, validation {-v:.4f}")
print("PICK", sorted(rows, reverse=True)[0][2], "(rule: highest dev success, tie -> lower validation error)")
PY
S=$(awk '/^PICK/{print $2}' data/eval/act_pick.txt)
echo "held-out $S start $(date)"
~/miniconda3/envs/gardener/bin/python scripts/eval_policy.py outputs/act/checkpoints/$S/pretrained_model > data/eval/act_heldout.txt 2>&1
grep -E "^POLICY|^SUCCESS|amount in the|failures by type" data/eval/act_heldout.txt
echo "HELDOUT-DONE $(date)"
