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
- **Narrow can layout:** see the known gap in CLAUDE.md. All 50 held-out scenes come from that narrower range.

Full reports: `docs/eval/act_heldout.txt`, `docs/eval/expert_heldout.txt`.

