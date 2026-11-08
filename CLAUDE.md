# iCub Little Gardener: project context for Claude

## What this project is
A personal robot-learning simulation. A simulated **iCub humanoid** grasps a small watering can with its
**right five-fingered hand** and waters plants (water simulated as particles). On top of the simulation:
1. a **scripted expert** (cuRobo motion planning + scripted grasp/pour) generates demonstrations,
2. demonstrations are converted to **LeRobot** format,
3. an **ACT** imitation-learning policy is trained, and **SmolVLA** is fine-tuned on language
   instructions like "water the left plant, just a little".

The full specification is the project book: `docs/iCub_Little_Gardener_Project_Book.pdf`
(chapters on architecture, task spec, randomisation ranges and acceptance criteria). Read the relevant
chapter before designing a new stage.

## Who you are working with
- Karima, a 4th-year robotics/ML engineering student. Strong in C++, Python, embedded; learning
  simulation physics and robot learning through this project.
- Explain choices in plain words, and say *why*, not only *what*. Short, clear answers.
- Working style for this project: **measure first, then change one thing.** Before a fix, write or run a
  check that shows the problem; after the fix, run the same check to show it is gone.

## Hardware and environment
- Laptop: i5-12500H, 16 GB RAM, **RTX 3050 Laptop GPU with only 4 GB VRAM**, Ubuntu 22.04 (dual boot).
- Keep GPU memory small: small batches, no huge particle counts on the laptop. Heavy training goes to
  free cloud GPUs (Kaggle, Lightning AI).
- Python environment: conda env `gardener` (Python 3.10), threepp 2026.10.11. TODO (Karima: confirm how
  threepp was installed).

## Stack
threepp (3D scene, rendering, PhysX bindings, URDF articulation) · NVIDIA PhysX · NVIDIA Warp (water
particles) · NVIDIA cuRobo (motion planning) · PyTorch · Hugging Face LeRobot (datasets, ACT, SmolVLA).

## Layout
```
gardener/sim/robot.py          # ICub class: loads the gardener URDF in PhysX, PD position drives, control by joint name
scripts/make_gardener_urdf.py  # original iCub URDF -> gardener.urdf (locks unused joints, lifts robot on a `world` link)
scripts/fit_hand_colliders.py  # gardener.urdf -> gardener_hand.urdf (capsule finger colliders, oriented palm box)
scripts/view_icub.py           # viewer / headless test of an arm + hand sequence
configs/poses.yaml             # named arm poses (DEGREES) and the demo sequence
docs/                          # project book
```
Robot assets live outside the repo in `~/icub-assets/iCub/robots/iCubGazeboV2_10/`
(`model_abs.urdf` = original with absolute mesh paths; never modify it).

## Common commands
```bash
# 1. Build the gardener URDF from the original
python scripts/make_gardener_urdf.py \
  --src ~/icub-assets/iCub/robots/iCubGazeboV2_10/model_abs.urdf \
  --out ~/icub-assets/iCub/robots/iCubGazeboV2_10/gardener.urdf --height 0.6

# 2. Fit grasp-friendly hand colliders
python scripts/fit_hand_colliders.py \
  --src ~/icub-assets/iCub/robots/iCubGazeboV2_10/gardener.urdf \
  --out ~/icub-assets/iCub/robots/iCubGazeboV2_10/gardener_hand.urdf

# 3. Watch / test
python scripts/view_icub.py                # window
python scripts/view_icub.py --test         # headless: prints max joint error and speed per phase
python scripts/view_icub.py --colliders    # draw collision shapes as yellow wireframes
```

## Conventions
- World is **z-up**. Joints in **radians** in code, **degrees** in `configs/poses.yaml`.
- The articulation has 85 joints but only **30 move**: torso (3), right arm (7), right hand (20:
  `r_hand_{thumb,index,middle,ring,little}_{0..3}_joint`). The other 45 are "locked" with near-zero
  limits (±1e-4 rad) and always get target 0.
- Generated URDFs are always written to a new file; source URDFs are never edited in place.

## Things we learned the hard way (PhysX / threepp gotchas)
- threepp turns each link's **collision mesh into a box (its bounding box)**, so fingers were blocks and
  the palm one big block. That is why `fit_hand_colliders.py` exists.
- A URDF `<cylinder>` collision is built by threepp as a **capsule along the link's z axis**. Finger
  capsules are fitted with PCA; they match the original geometry within about 1 mm.
- In PhysX, an articulation **link's pose is its centre-of-mass (COM) frame**, not the URDF link frame.
  Convert before comparing positions with URDF frames.
- Drive limits: check how `drive_limits_are_forces` is set. Finger drives once acted far stronger than
  the values we set; always verify actual drive force/stiffness against intended values.
- There was a **hidden collision shape in the palm** that blocked grasps. Always inspect colliders with
  `--colliders` before blaming the controller.

## Current status (as of November 2026)
- Robot loads and holds poses; arm + hand sequence runs (`view_icub.py --test` reports errors/speeds).
- Hand colliders fitted (capsule fingers, oriented palm box).
- **Grasping is not solved yet.**

## Next steps (in this order)
1. **Fix the palm box**: check its size/position/orientation against the palm mesh in `--colliders`
   view; make sure it does not stick out past the inner palm surface or overlap the finger bases.
2. **Proper contact tests**: small headless tests that close the hand on simple objects (box, cylinder
   the size of the can handle) and report contact points, contact forces and whether the object slips.
3. **Rerun the grasp search** for the watering-can handle with the fixed colliders, and record results.

## Rules for working in this repo
- Run `python scripts/view_icub.py --test` after any change to the robot, URDF scripts or drives, and
  compare with the numbers before the change.
- Prefer small, readable scripts with a docstring that says what it does and how to run it.
- Do not install new large dependencies without asking.
- Commit working states with clear messages; do not commit generated URDFs or large assets.
