# Results

## ACT + task ID (first policy), 2026-11-10

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
Held-out run: 2026-11-10 04:04-04:37, run once.

On dev, 80k never chose the wrong pot (0/50). Its weak point is "just a little": only 3 of 16 of those pours landed in band.

### Held-out: ACT + task ID next to the expert (seeds 5000-5049, same scenes, same scoring)

| | Expert | ACT + task ID (80k) |
|---|---|---|
| **Success rate** | 50/50 = 100% (93-100%) | **41/50 = 82% (69-90%)** |
| Water the left plant. | 9/9 | 5/9 |
| Water the right plant. | 9/9 | 7/9 |
| Left plant, just a little | 8/8 | 6/8 |
| Right plant, just a little | 8/8 | 8/8 |
| Left plant, a lot | 8/8 | 7/8 |
| Right plant, a lot | 8/8 | 8/8 |
| Wrong pot | 0/50 | 0/50 |
| **Amount in band**, all 50 scenes | 50/50 = 100% (93-100%) | 19/50 = 38% (26-52%) |
| Amount in band, successful episodes only | 50/50 | 15/41 |
| **Water in target**, % of what was poured: median, lowest, successful episodes | 100, 90 | 98, 75 |
| Water in target, mean over all 50 | 98 | 84 |
| Spill: median, worst | 0%, 10% | 5%, 97% (seed 5038) |
| **Collisions**, episodes with a collision among their failure reasons | 0 | 2: seed 5000, forearm on the can's handle 4 N; seed 5038, forearm on a pot or the table 15 N |
| Collisions as the first failure reason | 0 | 0 (both episodes had already failed for an earlier reason) |
| **Episode time**, mean of successful episodes | 42.8 s | 40.1 s |
| Largest joint target jump | 2.00 deg/frame | 2.00 deg/frame |

**Failures (9), by first reason:**
- 7 times the water missed the pot.
- 1 time nothing was poured: seed 5000 never lifted the can.
- 1 timeout: seed 5016, "a lot". It poured 94% into the target but was still going at 45 s.

**All failure reasons** (an episode can have several; rebuilt from the saved report with
`eval_expert.py --from-report`, no new run; `docs/eval/act_heldout_all_reasons.txt`):
- water missed the pot: 7 episodes;
- timeout: 2 (seed 5016, and seed 5000, which also poured nothing);
- collision: 2 (seed 5000 and seed 5038, which also missed the pot);
- nothing poured: 1.

Seed 5000 fails three ways: nothing poured, timeout and collision. The success rate does not change: 41/50.

The worst instruction is "Water the left plant." at 5/9.

**What it means:**
- **Pot choice works:** the policy never picked the wrong pot. The five task numbers are enough for it.
- **Amount works poorly:** only 38% of pours land in the instruction's band, against 100% for the expert. "Just a little" is the worst case, with 1 of 16 in band.
- **Aim is the main failure:** misses spill out of the pot. The policy lifts the can and pours on the right side, but its spout ends up off the rim more often than the expert's.
- **82% on held-out vs 78% on dev:** this is within the noise of 50 scenes, so it doesn't show the policy got better.

**Not like-for-like with the expert:**
- **Episode end:** a policy episode ends when the can is back upright on the table with the hand open for 1 s. The expert's ends after its retract. So the episode times are not measured the same way.
- **Speed limit:** the 1 deg/step limit on joint targets applies to the policy too.
- **Settings:** the policy ran with ACT's default of 100 actions per prediction and no temporal ensembling. Neither was tuned.
- **Narrow can layout:** can layouts where the can or the open hand would meet a pot are redrawn, so 41 of the 50 evaluation scenes have the can 20 mm or more towards iCub. All 50 held-out scenes come from that narrower range, not the book's full +-8 cm.

Full reports: `docs/eval/act_heldout.txt`, `docs/eval/expert_heldout.txt`.


### Evaluation settings for ACT + task ID (dev seeds, 2026-11-20)

The 80k checkpoint on the 50 dev seeds (0-49), with four settings for how often it asks the policy for new
actions. The run started at 21:55 and ended at 23:42. Held-out was not run: the default setting won, and it has
already been run on held-out once (41/50 above).

| Setting | Success (95% interval) | Amount in band, all 50 | Aim misses | Timeouts | Can not upright |
|---|---|---|---|---|---|
| **Default: every 100 frames (4 s)** | **39/50 = 78% (65-87%)** | **27/50** | **8** | 3 | 2 |
| Every 25 frames (1 s) | 32/50 = 64% (50-76%) | 27/50 | 11 | 11 | 6 |
| Every 50 frames (2 s) | 36/50 = 72% (58-83%) | 25/50 | 9 | 9 | 6 |
| Temporal ensembling 0.01 (every frame) | 39/50 = 78% (65-87%) | 20/50 | 9 | 3 | 2 |

- **How the columns are counted:** aim misses, timeouts and can not upright are episodes with that reason among
  *all* their failure reasons, so one episode can count in several columns.
- **Source of the default row:** its earlier report, rebuilt with `eval_expert.py --from-report`, with no new run.

**Pick: the default.** This follows the rule fixed before the run: highest dev success, with a tie keeping the
default. Ensembling ties it on success, but is worse on the amount (20/50 against 27/50 in band).

**What it means:**
- Asking more often did not fix aim or amount. Re-planning every 1-2 s made the policy hesitate: the timeouts
  went from 3 to 9-11, and it ended with the can tilted more often.
- Ensembling smooths the motion but blurs the end of the pour, so fewer pours land in the right band.
- The weak points come from what the policy learned, not from how often it is asked. The next step is
  training: train longer, or on more demonstrations.
- Ensembling and the default fail on different scenes: 5 seeds only the default fails (15, 16, 25, 29, 43) and
  5 only ensembling fails (6, 23, 36, 38, 42). With 50 scenes, the tie is not a real difference.

**Reports:** `docs/eval/act_dev_080000_{n25,n50,ens}.txt`, `act_dev_080000_all_reasons.txt`, `act_settings_pick.txt`.
Run script: `docs/eval/settings.sh`. Each seed is saved as it finishes (`--progress`), so a stopped run resumes.


## Dataset v2 (400 demonstrations), 2026-11-22

**Recording:** 200 more expert demonstrations, seeds 1204-1409. 200 of 206 attempts succeeded (97%; v1: 200 of 204).
- No seed is a dev seed (0-49) or a held-out seed (5000-5049), and none reuses v1's 1000-1203.
- Dataset `Kimaa6/icub_gardener_v2` (on the Hugging Face Hub since 2026-11-22, public since 2026-11-26, version tag v3.0; all 19 files checked against the local copy, same sizes). Episodes 0-199 are v1's, unchanged.
- **Split** (`configs/act_split_v2.json`): 360 training and 40 validation episodes, none in both. The 20 v1
  validation episodes stay in validation, plus 20 new ones balanced by instruction.

**Known video issue: 11 single-frame AV1 encoding glitches in the wrist-camera video.**
- **What:** a rectangle in the top right of the wrist view shows grey instead of the table, on one frame, during the
  pour.
- **Seeds (episode, frame):** 1220 (216, 581), 1230 (225, 615), 1244 (239, 571), 1290 (283, 621), 1304 (297, 575),
  1328 (321, 577), 1332 (325, 629), 1346 (338, 583), 1404 (394, 619 and 621), 1406 (396, 581).
- **The raw recorded frames are clean** in that region, so it is the video encoding, not the simulation.
- **None of these episodes is in validation.**
- **Scale:** 11 of 196,596 wrist frames (0.006%).
- Also one smaller 6-block patch in the eye camera: seed 1314 (episode 307, frame 681), raw clean.
- **Accepted as they are**; not re-encoded.
- **Limits of the scan:** it covered every frame of the new episodes 200-399 only. A few glitches fainter than its
  cut-off may remain.

Scan method and table: `docs/eval/v2_video_scan.txt`. Example: `docs/media/2026-11-22_wrist_encoding_glitch.png`.

## ACT + task ID retrained on dataset v2, dev seeds, 2026-11-23

**Training** (Kaggle, one T4): same policy settings as v1, 80k steps, batch 8, 360 training episodes.
- Validation error (40 validation episodes): 80k 0.0851, 70k 0.0857, 60k 0.0900. v1's best was 0.0986, but on its
  own 20 validation episodes, so the two are not directly comparable.
- Overfitting is much smaller: validation/training ratio 1.38 at 80k (v1: 2.33). Files: `docs/eval/act_v2_val.txt`.

**Dev seeds 0-49, default settings, exactly as v1 was evaluated:**

| Checkpoint | Dev success | Amount in band | Water missed the pot | Validation |
|---|---|---|---|---|
| v2 80k | 27/50 | 20/27 | 21 | 0.0851 |
| v2 70k | 26/50 | 18/26 | 19 | 0.0857 |
| **v2 60k (pick)** | **32/50** | 19/32 | 17 | 0.0900 |
| v1 80k (the v1 pick) | 39/50 | 23/39 | 10 | 0.0986 (v1 split) |

**v2 is worse than v1 on dev, although its validation error is lower.** What was checked:
- **The misses are pours beside the pot:** in the failed episodes, on average 60-77% of the poured water lands
  on the table. The right pot is chosen every time (no wrong-pot pours).
- **They depend on the can's yaw.** Dev successes by can yaw (60k vs v1): -25..-10 deg 3/12 vs 7/12;
  -10..0 6/11 vs 8/11; 0..10 8/11 vs 8/11; 10..25 15/16 vs 16/16.
- **Not a coverage gap:** the 200 new demos have more negative-yaw cans than v1's (111 below 0 deg vs 100).
- **Not worse demos:** in every yaw range, the new demos put 97-99% of the water in the pot, as v1's do, with the
  same drag and episode length.
- **Not the eval setup:** v1's 80k checkpoint, re-run on the same dev seeds on 2026-11-23, scores 39/50 again,
  with every one of the 50 seeds the same as on 2026-11-20 (`docs/eval/act_v1_recheck_080000.txt`). The two
  checkpoints' policy configs are identical. The v2 eval ran slower (0.40-0.43 x real time; the v1 recheck 0.55),
  but speed does not change the simulated result.
- **So the v2 policy itself is worse at negative yaw.** Two possible causes, not yet tested:
  1. **Half the passes over each demo:** the same 80k steps x batch 8 over 360 episodes instead of 180.
  2. **Run-to-run luck:** each version was trained only once.
- **Held-out (5000-5049), run once on the v2 pick (60k), 2026-11-23: 23/50 = 46% (95% interval 33-60%)**; v1: 41/50.
  - Amount in band: 8/23 (v1: 15/41).
  - Water missed the pot: 24 (v1: 7).
  - By can yaw, v2 vs v1: -25..-10 deg 3/15 vs 13/15; -10..0 8/13 vs 12/13; 0..10 6/10 vs 10/10; 10..25 6/12 vs 6/12.
  - The worst loss is again at strong negative yaw.
  - **v1 stays the project's best policy.** File: `docs/eval/act_v2_heldout.txt`.

Files: `docs/eval/act_v2_dev_{060000,070000,080000}.txt`, `docs/eval/act_v2_pick.txt`.

### Why is v2 worse at negative yaw? Two offline checks, 2026-11-23 (no simulation)

**1. Is the data different? No.** Per demo, from the measured joints and forward kinematics
(`scripts/compare_pour_demos.py`, output `docs/eval/pour_demos.txt`):
- **Can yaw:** v1's 200 demos have mean +0.8 deg, sd 14.9; the new 200 have mean -1.4 deg, sd 14.5.
  Below -10 deg: 59 vs 67 demos.
- **The pours match in every yaw range:**
  - same pour heading: mean about -90 deg, sd about 24 deg;
  - same furthest tilt: 75-79 deg;
  - same start (about 20.0 s) and length (about 4.4 s);
  - the same spread of the joints at the furthest tilt.
- **Split by target pot, the expert picks the same pour headings, in the same proportions, in both groups.** For
  example, right pot at -25..-10 deg: -120 deg in 25 of 25 (v1) and 34 of 35 (new).
- **The expert code did not change between the recordings:** only the failure logging and the video queue did.
- **One thing both groups share:** right pot, can yaw -25..-10 deg is the only case where the arm joints vary a
  lot at the same heading. The shoulder spread is 36-41 deg there, against 2-14 deg elsewhere. So the expert reaches
  that pour with more than one arm shape. It is equally true in v1's demos, so it does not explain v1 vs v2, but it
  is a hard case for any policy.

**2. Does v2 predict the expert's actions worse at negative yaw? No.** Error of the arm and torso actions on the
40 validation episodes, which neither policy trained on (`scripts/action_error_by_yaw.py`, output
`docs/eval/action_error_by_yaw.txt`). Measured in degrees, over the 100-step chunk, for every 10th frame:

| Can yaw | Episodes | Pour: v1 80k | Pour: v2 60k | Rest of the episode: v1 | Rest: v2 |
|---|---|---|---|---|---|
| -25..-10 | 15 | 2.45 | 2.49 | 1.80 | 1.60 |
| -10..0 | 7 | 2.37 | 2.32 | 1.62 | 1.44 |
| 0..10 | 9 | 3.00 | 2.69 | 1.50 | 1.42 |
| 10..25 | 9 | 2.29 | 2.74 | 1.32 | 1.24 |

- **On recorded demos, v2 is as good as v1 at negative yaw**, and slightly better outside the pour.
- **So the gap only appears when the policy drives the robot itself.** Its own small errors change what it sees next,
  and v2 drifts off where v1 recovers. Offline error cannot show this.
- **Training longer (160k steps) is therefore not a well-supported fix:** v2 already fits the demos as well as v1.
- **Limits:**
  - 7-15 validation episodes per yaw range;
  - a joint-angle error is not the spout's position error over the pot.
