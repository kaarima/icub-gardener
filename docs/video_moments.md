# Video moments

Moments worth filming: fails, fixes, first successes, big numbers. Each one has the date, what happened, the commit
and the seed or command that brings it back, so it can be re-rendered later.

**How to replay a moment:** run `git switch --detach <commit>` (then `git switch garden-scene` to come back), rebuild
the hand URDF if the commit changed `fit_hand_colliders.py` (step 2 in CLAUDE.md), then run the command.
`scripts/watch_expert.py` opens a live window for any expert seed (from commit e947c75).
Before commit 34b7ddd, iCub is drawn as its collision shapes, not its real meshes.

**Which instruction a seed gets:** the order is
`[left, right, left little, right little, left lot, right lot]`.
- Dev seeds 0-49 and recording seeds 1000+ use `list[seed % 6]`.
- Held-out seeds 5000-5049 use `list[(seed - 5000) % 6]`.

Full instruction texts: "Water the left plant.", "Water the right plant.",
"Give the left/right plant just a little water.", "Give the left/right plant a lot of water."

| Date | Moment | Commit | Seed / command to reproduce |
|---|---|---|---|
| 2026-10-13 | First look at the gardener iCub: 30 moving joints, robot lifted 0.6 m, 7 hidden meshes restored | 97f26e1 | `python scripts/view_icub.py` (screenshot: `docs/media/2026-10-13_icub_gardener_urdf.png`) |
| 2026-10-13 | **Fail:** the hand can't grasp. Every finger is a box, and a phantom sphere sits in the palm | d74debf (before the fix) | `python scripts/view_icub.py --colliders` |
| 2026-10-13 | Fix: capsule fingers fitted to the real meshes, within 1 mm | 54b55ff | `python scripts/view_icub.py --colliders` |
| 2026-11-08 | Thumb does touch after all. A free bar gripped, then gravity on: 0.0 mm slip | 8c31be7 | `python scripts/test_grasp_contacts.py --mode dynamic` |
| 2026-11-08 | First real lift of a can off a table: 16/16 pass once the timestep was found to cause the arm drift | 001babf | `python scripts/test_can_lift.py` |
| 2026-11-08 | **Fail → fix:** the forearm's box hits any table. As a capsule, the arm reaches over it | 0bf943b | `python scripts/test_can_lift.py`; before: 001babf with `view_icub.py --colliders` |
| 2026-11-08 | Physics settings: 2 ms/12 iterations, the can slips out (2/8); 2 ms/64 iterations holds (8/8) | 0bf943b | `python scripts/test_can_lift.py --solver-iters 12` vs `--solver-iters 64` |
| 2026-11-08 | **First success:** full reach, approach and thumb-last grasp, 27/27 over ±25° can rotation | 3717f42 | `python scripts/test_reach_grasp.py` |
| 2026-11-08 | First water: particles pour from a can moved by script. Tilt sets the amount: 50° 21%, 80° 65% | ff69a05 | `python scripts/test_pour.py --tilt 50` / `--tilt 80` |
| 2026-11-08 | First full episode from an instruction | e7045e7 | `python scripts/run_expert.py --seed 0 --instruction "Water the left plant."` |
| 2026-11-08 | **The fake 100%:** 50/50, but pots had no collider, so the can and arm passed through them | 088d8b4 | `python scripts/eval_expert.py` (any seed: watch the can go through a pot) |
| 2026-11-08 | **Big drop:** pots made solid, and the expert falls to 5/50 = 10% | 6935a19 | `python scripts/eval_expert.py` |
| 2026-11-08 | Fix: carry the can above the rims, then descend. 10% → 78% | eede5d6 | `python scripts/eval_expert.py` |
| 2026-11-08 | **Fail:** the forearm hits the table during release and retract | af831d2 | seeds 9, 29, 38, 47 (seed 48: can-body contact), e.g. `run_expert.py --seed 29 --instruction "Give the right plant a lot of water." --dump-contacts c.json` |
| 2026-11-08 | Fix: lift straight up before backing away. 78% → 84% | dd18582 | same seeds as above |
| 2026-11-09 | "A lot" pours finish within 45 s and land in band: dev 46/50 | fd05a0e | `python scripts/eval_expert.py` |
| 2026-11-09 | **Fail:** held-out 46/50, all 4 failures are the can body hitting a rim during the pour | fd05a0e | `python scripts/eval_expert.py --first-seed 5000` (failing seeds not noted; read the failure column) |
| 2026-11-09 | **Fail:** water passes through the can's walls when the fingers let go (7 mm lurch in one frame) | bab9aaf (before the fix) | `python scripts/eval_expert.py` (seed not noted: look for water lost on release) |
| 2026-11-09 | Fix: only lurches are spread over the water's substeps (doing it for every move sloshed water out in 29/50) | a09039f | `python scripts/eval_expert.py` |
| 2026-11-09 | **Big number:** expert 50/50 on dev and 50/50 on held-out, amount in band 50/50 | e315236 | `python scripts/eval_expert.py --first-seed 5000` |
| 2026-11-09 | **Fail → fix:** the arm snaps up to 35°/frame at the start of the return; after the fix it's at most 2.00°/frame | 9d9f2a6 (before), 737ca44 (after) | `python scripts/eval_expert.py` (worst seed not noted: look at the jump column) |
| 2026-11-09 | iCub gets its real look: real meshes, grey not red, plants, wrist camera. The eye camera first sat inside the skull | 34b7ddd | `python scripts/watch_expert.py`; images: `docs/media/2026-11-09_real_icub_sheet.png`, `..._wrist_camera_sheet.png` |
| 2026-11-09 | **Big number:** first dataset, 200 episodes from 204 attempts (98%) | eb2b2b3 | the failed seeds: 1015 (water missed), 1056 and 1091 (collision), 1103 (no place plan) |
| 2026-11-09 | **Fail → fix:** replaying saved actions drifted 12.5°; with the target set every step it drifts 1.4°, and 5/5 replays succeed | 591fb7d | `python scripts/replay_check.py --count 5` |
| 2026-11-09 | ACT smoke test: training loss 15.9 → 2.4 in 600 steps | a57e308 | `SMOKE=1 bash scripts/train_act.sh` |
| 2026-11-10 | First full ACT + task ID training, 80k steps overnight: loss 4.7 → 0.056. Best validation error 0.099 at 80k, but validation is 2.3× training (it overfits) | a57e308 | `bash scripts/train_act.sh` (curve: `docs/media/act_training_curve.png`, from `python scripts/plot_training.py`) |
| 2026-11-10 | **First learned policy:** ACT + task ID waters the right plant in 41/50 held-out scenes (82%) and never picks the wrong pot. Its pours often miss: seed 5038 spills 97% | 3fdc561 | `python scripts/run_expert.py --policy outputs/act/checkpoints/080000/pretrained_model --seed 5038 --instruction "Give the left plant just a little water."` (checkpoint on disk, not in git); a success: seed 5001 |
