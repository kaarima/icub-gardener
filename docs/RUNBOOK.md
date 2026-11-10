# Runbook: running iCub Little Gardener yourself

This guide runs the whole project in order, from a fresh terminal. Each step says:
- what it does,
- the command and what its flags mean,
- how long it takes,
- what you should see,
- how to know it worked,
- what to do if it fails.

Times are measured on the laptop (i5-12500H, RTX 3050 4 GB) unless marked *estimate*.

Every command runs from the repo folder: `cd ~/icub-gardener`.

---

## 0. The rules (read these first)

1. **One change per run.** Change one thing, rerun the 50 dev seeds, and compare before and after in a table.
   If two things change at once, you can't tell which one helped.
2. **Tune on dev, report on held-out once.**
   - **Dev seeds (0-49):** tune and pick on these as much as you like.
   - **Held-out seeds (5000-5049):** run once, at the end, on the final choice. Running them again after
     looking at the result turns them into dev seeds, and the number stops being honest.
   - **Recording seeds (1000 and up):** used only to make demonstrations.
3. **Commit after each step that works.** Use a clear message and push to the working branch (now
   `garden-scene`). Merge to `main` only when a stage is finished.
   - Never commit `data/`, `outputs/` or generated URDFs. They are in `.gitignore`.
   - To keep a result, copy its text report into `docs/eval/`.
4. **Measure first, then change.** Before a fix, show the problem with a number. After the fix, run the same
   check again.
5. **Write it down.**
   - Every result goes in `docs/log.md`, one dated line.
   - Final numbers go in `docs/results.md`.
   - Moments worth filming go in `docs/video_moments.md`: the date, the commit and the seed.

---

## 1. Environments and keeping the laptop awake

**What it does:** the project uses two conda environments, so that training libraries can't break the simulation.

| Env | Python | Holds | Used for |
|---|---|---|---|
| `gardener` | 3.10 | threepp (PhysX), Warp, numpy | simulation: running, recording, evaluating |
| `gardener-train` | | LeRobot 0.4.4, torch 2.10 | converting the dataset, training, the policy server |

The two can't be one env: installing LeRobot into `gardener` would downgrade torch and change its CUDA bindings.

```bash
conda activate gardener                       # for every sim command
# training commands, without activating:
env -u PYTHONPATH ~/miniconda3/envs/gardener-train/bin/python scripts/<script>.py
```

- `env -u PYTHONPATH` runs the command with `PYTHONPATH` removed. Your ROS setup puts old packages on
  `PYTHONPATH`, and LeRobot would import them by mistake.
- **Check it worked:** `python -c "import threepp, warp; print(threepp.__file__)"` in `gardener` prints a
  path inside `~/miniconda3/envs/gardener`. Trust `module.__file__`, not `pip list`, which ROS also confuses.
- **If an import fails:**
  1. First run `echo $PYTHONPATH`. If it shows ROS paths, use `env -u PYTHONPATH`.
  2. Check you are in the right env.

**Keeping the laptop awake (any run over 10 minutes):**

```bash
cat /sys/class/power_supply/AC/online       # must print 1 (charger in). Long runs on battery die halfway.
setsid nohup systemd-inhibit --what=sleep:idle:handle-lid-switch --why="gardener run" \
    bash data/eval/my_run.sh > data/eval/my_run.txt 2>&1 < /dev/null &
```

- `systemd-inhibit --what=sleep:idle:handle-lid-switch`: no sleep, no idle suspend, and closing the lid
  doesn't suspend, for as long as the command runs.
- `setsid nohup ... &`: the run keeps going if you close the terminal or VS Code.
- `> file 2>&1`: all output goes to a file. Follow it with `tail -f data/eval/my_run.txt`.
- `< /dev/null`: the run never waits for keyboard input.
- **Check it worked:** `systemd-inhibit --list` shows your "gardener run" line.
- **On a desk:** you can close the lid, but leave the vents clear, because training heats the GPU.
- **Start every run script with an AC check, so it refuses to start on battery.** See `docs/eval/overnight.sh`,
  which also has a watchdog that stops cleanly if the charger is pulled out.
- **Stopping a run:** use a bracketed pattern, `pkill -f "[e]val_expert.py"`. Without the brackets, `pkill`
  also matches its own shell and kills it.

---

## 2. Rebuild the URDFs and colliders

**What it does:**
- Makes `gardener.urdf` from iCub's original URDF. It locks the 45 joints we don't use and lifts the robot so
  its waist is 0.6 m above the floor.
- Then makes `gardener_hand.urdf`, with capsule fingers, a fitted palm box and a capsule forearm.

The original `model_abs.urdf` is never changed.

```bash
conda activate gardener
A=~/icub-assets/iCub/robots/iCubGazeboV2_10
python scripts/make_gardener_urdf.py --src $A/model_abs.urdf --out $A/gardener.urdf --height 0.6
python scripts/fit_hand_colliders.py --src $A/gardener.urdf --out $A/gardener_hand.urdf
python scripts/view_icub.py --test
```

**Flags:**
- `--src` is the input URDF. `--out` is a **new** file to write; never point it at the source.
- `--height` is the waist height above the floor, in metres.
- `make_gardener_urdf.py --lock-mode fixed` would lock joints as fixed joints instead of using tiny limits.
  We use the default, `limits`.

**Time:** under a minute each. `view_icub.py --test` runs one sequence of moves *(estimate: under a minute)*.

**Expect:**
- `fit_hand_colliders.py` prints each finger's capsule. The fit matches the real mesh within about 1 mm.
- `view_icub.py --test` prints the phase every second, with `err` (largest gap between target and actual
  joint angle, in degrees) and `speed` (largest joint speed, in degrees per second).

**Check it worked:**
- `err` should be small and `speed` near 0 during the "hold" phases.
- Compare these numbers with the run before your change, as CLAUDE.md asks.
- `python scripts/view_icub.py --colliders` draws the collision shapes as yellow wireframes. The fingers
  should be capsules, not boxes.

**If it fails:** look at the colliders before blaming the controller. A hidden palm shape once blocked every
grasp.

---

## 3. Run and watch one expert episode

**What it does:** the scripted expert does one full episode, for one seed and one instruction:
- reach the can;
- grasp it, with the fingers first and the thumb last;
- lift it;
- carry it above the pot rims;
- tilt to pour;
- tilt back and put the can down;
- let go and move the arm back.

The seed decides the scene: can position and turn, how full it is, the can's mass, and so on.

```bash
conda activate gardener
python scripts/watch_expert.py --seed 1 --instruction "Water the right plant."      # live window
python scripts/run_expert.py   --seed 0 --instruction "Water the left plant."       # no window, prints the metrics
```

**Flags:**
- `--seed`: which random scene.
- `--instruction`: one of the six texts:
  - "Water the left plant." / "Water the right plant."
  - "Give the left plant just a little water." / "Give the right plant just a little water."
  - "Give the left plant a lot of water." / "Give the right plant a lot of water."
- `watch_expert.py`: drag to orbit, scroll to zoom. `--test-frames N` runs N steps with no window, to test the
  script itself.
- `run_expert.py` options:
  - `--quiet` prints only the result.
  - `--can-yaw D` forces the can's turn to D degrees.
  - `--miss-pot-mm M` aims M mm off the pot, on purpose, to test spill detection.
  - `--dump-contacts f.json` saves the arm's pose at every contact.
  - `--record f.npz` saves the episode if it succeeds; `--fps` sets frames per second (default 25).
  - `--replay f.npz` plays back a recorded episode's actions instead of the expert.
  - `--policy CKPT` runs a trained policy instead of the expert; see step 9.

**Time:**
- About 5 s before the window opens, while it plans and lets the robot settle.
- The episode itself is about 36-43 s of simulated time, and the sim runs at about 1x real time.

**Expect:** `run_expert.py` ends with a line like
`RESULT: SUCCESS, reached end`, or `RESULT: FAILED: <reason>, reached <phase>`. Before it, it prints:
- drag (how far the can slid while the fingers closed);
- lift;
- poured % (share of the can's water that came out);
- target % (share of the poured water that landed in the right pot);
- spill;
- the largest joint-target jump;
- rim clearances per phase;
- any contacts.

**Check it worked:** `SUCCESS`, with no `contacts:` line, and target % at 90 or more for the expert.

**If it fails:** the reason after `FAILED:` tells you which part:
- **"no place plan"**: the IK couldn't find a pose. Try a different seed. Some layouts are too tight.
- **"collision"**: rerun with `--dump-contacts c.json` to see which body touched what, with how much force,
  and in which phase.

---

## 4. Evaluate the expert on 50 seeds and read the report

**What it does:** runs the expert on 50 scenes, one process per scene (PhysX allows only one world per
process), and prints one row per scene and then a summary.

```bash
conda activate gardener
python scripts/eval_expert.py --first-seed 0    --workers 4 | tee data/eval/expert_dev.txt      # dev
python scripts/eval_expert.py --first-seed 5000 --workers 4 | tee data/eval/expert_heldout.txt  # held-out, ONCE
```

**Flags:**
- `--first-seed`: 0 for dev, 5000 for held-out.
- `--episodes`: how many scenes; the default is 50.
- `--workers`: scenes run at the same time. 4 fits 16 GB of RAM for the expert; use 2 when a policy runs.
- `| tee file`: shows the output and also saves it.

The instruction cycles through the six texts: seed `first-seed + i` gets text number `i % 6`.

**Time:** about 30-35 min for 50 scenes with 4 workers. The run prints its speed, about 0.55x real time
overall.

**Reading the report:**
- **One row per seed:**
  - yaw, dx and dy: the can's turn and shift;
  - fill: how full the can starts;
  - drag and lift, in mm;
  - pour%, tgt% and spill;
  - jump: the largest joint-target change in one frame;
  - band: whether the poured amount is inside the instruction's band;
  - tilt: the can's tilt after it is put down;
  - place: placement error, in mm;
  - marg: wrist margin, in degrees;
  - spout: spout aim error;
  - lost: water lost before the pour;
  - failure: the reason, for failed episodes.
- **`SUCCESS: 50/50 = 100% (95% interval 93-100%)`:** the 95% interval is how sure 50 scenes make you.
  If two intervals overlap a lot, the difference may be luck.
- **Amount in the instruction's band:** "little" is 20-30% of the water, "normal" 40-60%, "a lot" 60-80%.
- **Spill, joint target jump, can height and rim clearance per phase:** quality checks. Clearance is how close
  the can came to a pot rim, in mm; negative means it touched.
- **By instruction:** success, amount in band and wrong pot for each of the six texts.
- **Failures by type:** the count of each failure reason.

**Check it worked:** the expert should get about 50/50 on both dev and held-out (`docs/eval/expert_heldout.txt`).

**If it fails:**
- **A worker crashed:** a seed then shows an error, not a result. Rerun that one seed with `run_expert.py` to
  see the full error.
- **Results look worse than before:** check `git status` for uncommitted changes.

---

## 5. Record demonstrations, convert to LeRobot, check, split

### 5a. Record

**What it does:** runs the expert on new seeds from 1000 up and keeps only successful episodes, each as a raw
`.npz` file. Each file holds three camera images per frame (eye, wrist and scene), the joint states, the
actions, and the instruction.

```bash
conda activate gardener
python scripts/record_episodes.py --count 200 --workers 2
```

**Flags:**
- `--count`: how many *successful* episodes you want in total. Episodes already in the folder count too.
- `--first-seed`: default 1000. Never use 0-49 or 5000-5049.
- `--workers`: 2, because rendering three cameras uses GPU memory.
- `--out`: the folder; default `data/raw`.

**Time:**
- About 80 s per episode for each worker, so 200 episodes take about 2.5 h.
- 200 episodes use about 5.4 GB of raw files.

**Expect:**
- One line per seed: `seed 1000 saved (76 s): ... | RESULT: SUCCESS`.
- Failed seeds are listed in `data/raw/failed_seeds.txt`.

**Check it worked:** about 98% of attempts are saved (200 of 204 last time).

**If it stops:** just run it again. It skips seeds that are already saved or already tried.

### 5b. Convert to LeRobot

**What it does:** writes the LeRobot dataset to `data/lerobot/gardener_v1`:
- the images become MP4 videos;
- each frame also stores the instruction as five numbers (`observation.environment_state`).

```bash
env -u PYTHONPATH ~/miniconda3/envs/gardener-train/bin/python scripts/to_lerobot.py
```

**Flags:**
- `--raw` and `--out`: the folders.
- `--repo-id`: the dataset's name. The default is `Kimaa6/icub_gardener_v1`, and it is not uploaded here.
- `--png`: the old, slower way, which writes PNG files first. Only use it to compare.

**Time:** about 26 min for 200 episodes. The result is about 1.4 GB.

**Check it worked:** the script stops with an error if the video encoder dropped a frame. A clean finish means
no frames were dropped.

**If it fails:** rerun it. It replaces the older copy.

### 5c. Check the dataset

**What it does:** compares every episode in the LeRobot dataset with its raw file:
- frame counts;
- the instruction;
- state and action values (they must match exactly);
- decoded image quality (PSNR);
- each episode's video span;
- the normalisation statistics.

```bash
env -u PYTHONPATH ~/miniconda3/envs/gardener-train/bin/python scripts/check_dataset.py --sheet data/check.png
```

**Flags:**
- `--sheet`: save a picture of frames from some episodes.
- `--images-per-episode`: how many frames per episode are decoded and compared; the default is 6.

**Time:** a few minutes.

**Expect:**
- mean, std and min..max for each of the 5 instruction numbers, 12 state values and 12 actions;
- episodes per instruction;
- the worst difference and the worst PSNR.

**Check it worked:**
- The last line is `RESULT: all checks passed`.
- The worst PSNR is about 30 dB or more (14.7 dB once meant broken video).
- No std is near 0.

**If it fails:**
- **Low PSNR or a bad video span:** convert again.
- **A std near 0:** that value never changes in the data. Look at the recorder.

### 5d. Split train and validation

**What it does:** puts 20 of the 200 episodes aside for validation, 3-4 per instruction, with a fixed seed so
it's repeatable.

```bash
env -u PYTHONPATH ~/miniconda3/envs/gardener-train/bin/python scripts/make_split.py --val 20
```

**Flags:**
- `--val`: number of validation episodes.
- `--seed`: changes which episodes are picked. Leave it at 0, so everyone uses the same split.

**Time:** seconds.

**Check it worked:** it prints the validation and training counts per instruction. Commit
`configs/act_split.json`.

---

## 6. Upload to Hugging Face (you, once)

**What it does:** uploads the dataset as a **private** repo, so Kaggle or Lightning can download it.

```bash
env -u PYTHONPATH ~/miniconda3/envs/gardener-train/bin/hf auth login       # paste a token with write access
env -u PYTHONPATH ~/miniconda3/envs/gardener-train/bin/python scripts/push_dataset.py
```

**Flags:** `--yes` skips the "upload? [y/N]" question. `--repo-id` defaults to `Kimaa6/icub_gardener_v1`.

**Time:** depends on your upload speed, for 1.4 GB.

**Check it worked:** on huggingface.co/datasets/Kimaa6/icub_gardener_v1 the repo shows **Private** and the
episode count is right.

**If it fails:**
- **"401" or "403":** log in again with a token that has *write* access.
- **The upload stops halfway:** run it again.

---

## 7. Train ACT

### How ACT works, in plain words

**What it sees, at each frame:**
- three camera images (eye, wrist and scene), each through a ResNet-18;
- 12 numbers for the robot: 3 torso and 7 arm joint angles, finger and thumb closure;
- 5 numbers for the instruction (task ID): left, right, a little, normal, a lot.

**What it predicts:** a **chunk** of the next 100 actions (4 s at 25 fps). Each action is 12 joint targets.
The sim plays the chunk, then asks again.

**How it trains:**
- A transformer turns the inputs into the chunk.
- During training, a small extra network (the VAE, a variational autoencoder) summarises the expert's real
  chunk into a "style" code. The loss is:
  - **L1**, the gap between the predicted and the real actions;
  - plus **KL**, which keeps that style code simple.
- At test time the style code is set to zero.

**Why chunks:** predicting 4 s at once keeps the motion smooth and stops small errors from adding up every
frame. The cost is that the policy is "blind" while a chunk runs.

**Why task ID:** plain ACT can't read text, so the instruction goes in as five numbers. Every result is
labelled **ACT + task ID** for that reason.

**Files:**
- `docs/training.md`: what ACT sees and the cloud setup.
- `gardener/data/instructions.py`: the five numbers.
- `scripts/train_act.sh`: the training command.
- `scripts/policy_server.py`: runs the policy.
- `gardener/policy/client.py`: connects the sim and the policy.

### 7a. On the laptop

```bash
cd ~/icub-gardener
SMOKE=1 PATH=~/miniconda3/envs/gardener-train/bin:$PATH env -u PYTHONPATH bash scripts/train_act.sh   # 1.5 min check
STEPS=80000 BATCH=8 SAVE=10000 LOG=500 OUT=outputs/act \
  PATH=~/miniconda3/envs/gardener-train/bin:$PATH env -u PYTHONPATH bash scripts/train_act.sh > data/eval/act_train.log 2>&1
```

Run the long one inside a run script with an AC check and `systemd-inhibit` (step 1). `docs/eval/overnight.sh`
is the one used for the first policy.

**Settings (environment variables):**
- `STEPS`: training steps.
- `BATCH`: samples per step. 8 fits in 4 GB.
- `SAVE`: save a checkpoint every this many steps.
- `LOG`: print the loss every this many steps.
- `OUT`: output folder.
- `WORKERS`: data-loading processes; default 4.
- `DEVICE`: `cuda` or `cpu`.
- `AMP=true`: mixed precision, which is faster and uses less memory.
- `SPLIT`: the split file.
- `DATASET`: the dataset folder.
- `SMOKE=1`: 200 steps at batch 4, written to `outputs/act_smoke`.

**Time:** 80k steps take about 4 h 20 min (5.4 steps/s), using 1.9 GB of the 4 GB of GPU memory.

**Don't use the GPU for anything else while it trains.**

**Expect:**
- A progress bar.
- Every `LOG` steps, a line with `loss:`. It falls from about 4.7 to about 0.05 over 80k steps.
- Checkpoints appear in `outputs/act/checkpoints/010000 ... 080000/pretrained_model`.

**Check it worked:** the loss falls and all 8 checkpoints exist.

**If it fails:**
- **Out of memory:** use `BATCH=4`.
- **It stopped halfway:** resume with
  `lerobot-train --config_path=outputs/act/checkpoints/last/pretrained_model/train_config.json --resume=true`.

### 7b. On Kaggle (free T4, 16 GB)

Follow `docs/training.md` section 3a.
- In short: a new notebook with GPU T4 and Internet on, your HF token stored as the secret `HF_TOKEN`, then
  `pip install lerobot==0.4.4`, `hf download Kimaa6/icub_gardener_v1 ...`, `git clone` and
  `DATASET=... OUT=/kaggle/working/act bash scripts/train_act.sh`.
- **Update the branch first:** `docs/training.md` clones the old `act-training` branch, which stops at commit
  f49e0bc and doesn't have the latest scripts. Clone `garden-scene`, or `main` after a merge.
- **Time:** not measured on a T4 *(estimate: similar to the laptop or faster)*. A Kaggle session lasts at
  most 12 h.
- **After training:** download `checkpoints/` from the Output tab into `outputs/act/` on the laptop.

### 7c. Plot the training curve

```bash
conda activate gardener
python scripts/plot_training.py --log data/eval/act_train.log --val data/eval/act_val.txt \
    --out docs/media/act_training_curve.png
```

**Flags:**
- `--log`: the training log.
- `--val`: the output of `val_loss.py` (step 8a). Run that first.
- `--out`: the image to write.
- `--title`: the chart title.

**Time:** seconds, on the CPU.

**Expect:**
- left: the training loss (log scale);
- right: validation and training action error at each checkpoint.

**Check it worked:** open the PNG.
- Validation should fall.
- If validation rises while training keeps falling, the policy is memorising. Pick an earlier checkpoint.

---

## 8. Pick a checkpoint, run held-out once, write results

### 8a. Validation error per checkpoint

**What it does:** for each checkpoint, measures the action error on the 20 validation episodes and on 20
training episodes. It is the mean absolute gap between the predicted and the recorded action chunks, with the
policy in evaluation mode.

```bash
env -u PYTHONPATH ~/miniconda3/envs/gardener-train/bin/python scripts/val_loss.py \
    outputs/act/checkpoints/0*/pretrained_model --batches 40 | tee data/eval/act_val.txt
```

**Flags:**
- `--batches`: batches of 16 random frames per set. More means steadier numbers, but slower.
- `--split`: the split file.
- `--device`: `cuda` or `cpu`.

**Time:** about 15 min for 8 checkpoints.

**Expect:** `...: validation 0.0986, training 0.0423, ratio 2.33` for each checkpoint.

### 8b. Dev seeds for the best 3

```bash
conda activate gardener
python scripts/eval_policy.py outputs/act/checkpoints/080000/pretrained_model --first-seed 0 | tee data/eval/act_dev_080000.txt
```

Do this for the 3 checkpoints with the lowest validation error.

**Flags:**
- `eval_policy.py CKPT`: runs `eval_expert.py` with the policy in place of the expert.
- Its defaults are `--first-seed 5000` (held-out) and `--workers 2`. `--first-seed 0` switches to dev.

**Time:** about 35 min per checkpoint.

**Expect:** the same report as the expert, starting with `POLICY: ACT + task ID`.

**The rule:** the highest dev success wins. If two tie, the lower validation error wins. Write the rule down
**before** you look at the results.

### 8c. Held-out, once

```bash
python scripts/eval_policy.py outputs/act/checkpoints/<picked>/pretrained_model | tee data/eval/act_heldout.txt
```

`docs/eval/heldout.sh` applies the rule automatically and then runs held-out once.

**Time:** about 33 min.

**Expect:** the report. The first policy got `SUCCESS: 41/50 = 82% (69-90%)`.

### 8d. Write the results

1. Copy the reports into `docs/eval/`.
2. Add a table to `docs/results.md`, with the expert and the policy side by side:
   - success rate with its interval;
   - success per instruction;
   - amount in band;
   - water in target;
   - collisions;
   - episode time.
3. Add one line to `docs/log.md` and a row to `docs/video_moments.md`.
4. Commit and push.

**Say what's not like-for-like:** a policy episode ends at the done rule (can upright, hand open for 1 s).
The expert's ends after its retract.

### How scoring works, in plain words

The policy and the expert are scored by the same code: the `score` function in
`gardener/expert/state_machine.py`, with thresholds in `configs/scene.yaml`. The policy runs through
`run_controller` in the same file.

**At the end of an episode, it counts where each water particle is:** in the can, in each pot, on the table,
or on the floor. From that:
- **poured** = particles that left the can ÷ all particles;
- **water in target** = particles in the right pot ÷ poured;
- **spill** = particles on the table or floor ÷ poured;
- **wrong pot** = more water in the other pot than in the right one;
- **amount in band** = poured is inside the instruction's band ("little" 20-30%, "normal" 40-60%,
  "a lot" 60-80%).

**Success needs all of these:**
1. Something was poured.
2. Water in target is at least **70%**.
3. The can is back on the table: within 1 cm of its starting height and tilted at most **10°**.
4. The episode took at most **45 s**.
5. **No collisions:** no robot part outside the hand touched anything, and the can didn't hit a pot or the table
   with more than 10 N (`MIN_SCENERY_FORCE`).

The amount band is reported, but it is **not** part of success.

**The failure reason is the first one that applies, in this order:**
1. nothing poured;
2. wrong pot;
3. water missed the pot;
4. can not upright;
5. timeout;
6. collision.

So an episode that missed the pot *and* touched a rim is listed as "water missed". The `contacts:` lines
in the report show the contacts anyway.

The 95% interval is the Wilson interval: with 50 scenes, 82% really means "somewhere between about 69% and
90%".

---

## 9. Evaluation settings: n_action_steps and ensembling

**What they do:**
- **`--n-action-steps N`:** play only the first N actions of each predicted chunk, then ask the policy again.
  - The default is the checkpoint's own setting: 100, the whole 4 s.
  - A smaller N, such as 20 or 25, reacts faster to what the cameras see. It may also be less smooth.
- **`--ensemble C`:** ask the policy **every frame**, and average all the chunks that predict the current
  frame. Older predictions get weight `exp(-C * age)`. The ACT paper uses C = 0.01.
  - Smoother and reacts fastest.
  - The policy runs 25 times per simulated second, so it is slower.

Neither setting needs retraining:

```bash
python scripts/eval_policy.py CKPT --first-seed 0 --n-action-steps 25 | tee data/eval/act_dev_n25.txt
python scripts/eval_policy.py CKPT --first-seed 0 --ensemble 0.01    | tee data/eval/act_dev_ens.txt
```

**Time:** about 35 min each, on the dev seeds. Ensembling may take longer *(estimate)*.

**Rule:** one setting per run, on **dev**. Compare with the dev result at the default setting (39/50 for 80k).
Only the winner goes to held-out, once, and the report says held-out has now been used twice.

**Why try them:** the first policy's failures are aim and amount, both decided at the end of the pour. A policy
that asks only every 4 s can't react in time to "stop pouring now".

---

## 10. Next: SmolVLA fine-tuning and evaluation (not built yet)

**What it is:** SmolVLA is a small vision-language-action model from Hugging Face. Unlike ACT, it **reads the
instruction text itself**, so it needs no task ID. The dataset already stores the text as each frame's
`task`, so the dataset doesn't change.

**Plan (these commands are a sketch, not tested here):**
1. Train on Kaggle or Lightning, not the laptop. SmolVLA has about 450M parameters, and fine-tuning won't fit
   in 4 GB.
   ```bash
   lerobot-train --policy.path=lerobot/smolvla_base --dataset.repo_id=Kimaa6/icub_gardener_v1 \
       --dataset.root=$DATASET --dataset.episodes=<train list from configs/act_split.json> \
       --batch_size=<fits the GPU> --steps=<to decide> --output_dir=$OUT --policy.push_to_hub=false
   ```
   Use the same 180/20 split, so the validation error compares with ACT's.
2. **Evaluate with the same scripts.**
   - `eval_policy.py` already labels a SmolVLA checkpoint "SmolVLA".
   - `policy_server.py` already sends the instruction text as the task.
   - Check first that the server runs it on the 4 GB GPU at batch 1. If not, run evaluation on the CPU
     *(slow)* or in the cloud.
3. **Same rules:** dev seeds to pick, held-out once, then a table next to the expert and ACT + task ID.
4. **Language test:** try instructions it never saw, like "give the left plant a tiny bit". This is the
   SmolVLA part of the project book.

Read the SmolVLA chapter of the project book before starting this stage.

---

## 11. Making video clips from video_moments.md

There's no clip script yet. Three ways, from easiest:

1. **Live window and screen recorder.** For any expert moment from commit e947c75 on:
   - `python scripts/watch_expert.py --seed S --instruction "..."`;
   - record the window with Ubuntu's recorder: `Ctrl+Shift+Alt+R` starts and stops; it stops by itself
     after 30 s by default;
   - or record it with OBS.
2. **An older moment.** Go back to its commit, rebuild the URDFs if that commit changed them (step 2), run
   the moment's command, and screen-record it. Then come back:
   ```bash
   git switch --detach <commit>
   # ... run and record ...
   git switch garden-scene
   ```
   Before 34b7ddd, iCub is drawn as its collision shapes, not its real meshes.
3. **Recorded episodes.** The dataset's MP4s are in
   `data/lerobot/gardener_v1/videos/observation.images.{scene,eye,wrist}/chunk-000/file-*.mp4`. Each file holds
   several episodes one after another. Each episode's start and end times are in `meta/episodes`. Cut a clip
   with `ffmpeg -ss START -to END -i file-000.mp4 -c copy clip.mp4`.

**Policy episodes:** `watch_expert.py` can't run a policy yet. Two ways for now:
- `run_expert.py --policy CKPT --record f.npz` saves the frames of a successful episode. Make a sheet with
  `episode_sheet.py`.
- Or add a `--policy` flag to `watch_expert.py`, the next small task.

**Also record the things you can't make again later:** your own reactions, and screenshots of live training.
The training curve itself can be redrawn from the log.

---

## Quick reference: where things are

| What | Where |
|---|---|
| Project spec | `docs/iCub_Little_Gardener_Project_Book.pdf` |
| Daily log | `docs/log.md` |
| Final numbers | `docs/results.md`, reports in `docs/eval/` |
| Moments to film | `docs/video_moments.md` |
| Scene, success thresholds, timeout | `configs/scene.yaml` |
| Expert and scoring | `gardener/expert/state_machine.py` |
| Simulation, robot, water | `gardener/sim/` |
| Recorder and cameras | `gardener/data/record.py` |
| Train/validation split | `configs/act_split.json` |
| Raw episodes, dataset, run outputs (not in git) | `data/raw`, `data/lerobot/gardener_v1`, `data/eval` |
| Checkpoints (not in git) | `outputs/act/checkpoints/<step>/pretrained_model` |
