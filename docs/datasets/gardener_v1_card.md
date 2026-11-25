Simulated demonstrations from the iCub Little Gardener project (https://github.com/kaarima/icub-gardener): the iCub
humanoid grasps a watering can with its right hand and waters the plant named in the instruction, such as "Give the
left plant just a little water". Everything is simulated (threepp with NVIDIA PhysX, particle water in NVIDIA Warp).
The demonstrations come from a scripted expert.

- **200 episodes, 196,646 frames, 25 fps**, 6 instructions (left or right plant; normal, "just a little" or "a lot").
- **Seeds 1000-1203:** only successful expert episodes are kept, 200 of 204 attempts.
- **Cameras:** eye, wrist and scene, 224 x 224 RGB, AV1 video.
- **State:** 3 torso and 7 arm joints (rad), finger and thumb closure. **Action:** 12 joint and closure targets.
- **Task code:** `observation.environment_state`, 5 numbers for the target plant and the amount, used by
  "ACT + task ID". SmolVLA ignores it and reads the instruction sentence instead.
- **Split used for training** (`configs/act_split.json` in the repository): 180 training and 20 validation episodes.
  ACT + task ID trained on this split succeeds in 41 of 50 held-out simulated scenes.

The 400-episode dataset `Kimaa6/icub_gardener_v2` holds these 200 episodes unchanged, plus 200 new ones.
Its video scan for AV1 encoding glitches covered only its new episodes, so this dataset has not been scanned.
