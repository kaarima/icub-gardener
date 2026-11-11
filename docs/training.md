# Training ACT

The dataset is recorded and checked on the laptop; ACT trains on a free cloud GPU; the trained policy comes
back to the laptop for evaluation in the simulator, which only runs here.

## What ACT sees

| Input | Shape | Note |
|---|---|---|
| `observation.images.eye`, `.wrist`, `.scene` | 224 x 224 RGB | ResNet-18, ImageNet weights |
| `observation.state` | 12 | 3 torso and 7 arm joints (rad, measured), finger and thumb closure (0-1, commanded) |
| `observation.environment_state` | 5 | the instruction as numbers: left, right, a little, normal, a lot |
| `action` (predicted) | 100 x 12 | the next 4 s of joint targets and closures, at 25 fps |

ACT cannot read text, so the instruction goes in as five numbers (`gardener/data/instructions.py`). Without
them every episode would look alike to it until the pour, and it could only guess the pot and the amount.
This is a departure from plain ACT, so every results table calls it **ACT + task ID** (eval_policy.py prints
the label). SmolVLA later reads the text itself.

## 1. On the laptop: split and smoke test

```bash
# gardener-train env, ROS PYTHONPATH cleared
env -u PYTHONPATH ~/miniconda3/envs/gardener-train/bin/python scripts/make_split.py --val 20   # -> configs/act_split.json
PATH=~/miniconda3/envs/gardener-train/bin:$PATH SMOKE=1 env -u PYTHONPATH bash scripts/train_act.sh
```

The smoke run does 200 steps at batch 4: the loss must fall and a checkpoint must appear. It is the book's
Phase 4 step 1. Commit `configs/act_split.json`, so the cloud run uses the same split.

## 2. Upload the dataset (you do this, once)

Hugging Face Hub, private (then Kaggle and Lightning can both download it):

```bash
env -u PYTHONPATH ~/miniconda3/envs/gardener-train/bin/hf auth login          # a token with write access
env -u PYTHONPATH ~/miniconda3/envs/gardener-train/bin/python scripts/push_dataset.py   # asks before uploading
```

On Kaggle, add your Hub token as a secret named `HF_TOKEN` and download it before training:

```bash
!pip install -q lerobot==0.4.4
import os; from kaggle_secrets import UserSecretsClient; os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
!hf download Kimaa6/icub_gardener_v1 --repo-type dataset --local-dir /kaggle/working/gardener_v1
```

and use `DATASET=/kaggle/working/gardener_v1` below. Or zip `data/lerobot/gardener_v1` and upload it as a
private Kaggle dataset instead. Nothing in this repo uploads anything without you running it.

## 3a. Kaggle (T4 x1 or P100, 16 GB, about 30 GPU hours a week, 12 h per session)

New notebook, Accelerator: GPU T4 x1, Internet: on (and Add Input: your dataset, if you used a Kaggle dataset). Cells:

```bash
!git clone -b main https://github.com/kaarima/icub-gardener.git
!pip install -q lerobot==0.4.4
!cd icub-gardener && DATASET=/kaggle/input/icub-gardener-v1/gardener_v1 OUT=/kaggle/working/act \
   STEPS=80000 BATCH=8 WORKERS=4 bash scripts/train_act.sh
```

If the repository is private, clone with a GitHub token stored as a Kaggle secret, not typed into the notebook.
Checkpoints land in `/kaggle/working/act/checkpoints/<step>/pretrained_model`: download them from the
notebook's Output tab. A session that stops can continue with
`lerobot-train --config_path=/kaggle/working/act/checkpoints/last/pretrained_model/train_config.json --resume=true`.

## 3b. Lightning AI (free credits, a real Linux machine with a terminal)

Start a Studio with a T4, then in its terminal:

```bash
git clone -b main https://github.com/kaarima/icub-gardener.git && cd icub-gardener
pip install lerobot==0.4.4
# copy the dataset in (upload the zip through the file browser, then unzip into data/lerobot/)
STEPS=80000 BATCH=8 bash scripts/train_act.sh
```

Switch the Studio back to CPU, or stop it, when training ends: credits are used while the GPU is on.

## How long

Not measured yet. Measure it: the first 1000 steps print their speed. 80,000 steps at batch 8 is about 3,600
passes over each training frame. If the T4 is too slow for one session, use `STEPS=40000`, then compare
checkpoints with the validation error before training longer.

## 4. Back on the laptop: pick a checkpoint, then evaluate

```bash
# validation vs training error per checkpoint (gardener-train env)
env -u PYTHONPATH ~/miniconda3/envs/gardener-train/bin/python scripts/val_loss.py outputs/act/checkpoints/*/pretrained_model

# 50 held-out seeds, same report as the expert (gardener env)
python scripts/eval_policy.py outputs/act/checkpoints/040000/pretrained_model | tee data/eval/act_heldout.txt
python scripts/eval_policy.py CKPT --ensemble 0.01          # temporal ensembling, as in the ACT paper
```

Choose the checkpoint on validation error and the dev seeds (`--first-seed 0`), then run held-out once.
The policy is scored exactly like the expert: `run_controller` in `gardener/expert/state_machine.py` runs it
in the same scenes, with the same contact checks and success rule, and counts the episode as done when the
can is back on the table, upright, with the hand open for 1 s.
