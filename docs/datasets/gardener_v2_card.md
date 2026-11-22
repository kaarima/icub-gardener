Simulated demonstrations from the iCub Little Gardener project (https://github.com/kaarima/icub-gardener): the iCub
humanoid grasps a watering can with its right hand and waters the plant named in the instruction, such as "Give the
left plant just a little water". Everything is simulated (threepp with NVIDIA PhysX, particle water in NVIDIA Warp).
The demonstrations come from a scripted expert.

- **400 episodes, 25 fps**, 6 instructions (left or right plant; normal, "just a little" or "a lot"), about 66 each.
- **Seeds 1000-1409:** episodes 0-199 are dataset v1 (`Kimaa6/icub_gardener_v1`) unchanged; episodes 200-399 are new.
  Only successful expert episodes are kept: 200 of 204 attempts for v1, 200 of 206 for the new ones.
- **Cameras:** eye, wrist and scene, 224 x 224 RGB, AV1 video.
- **State:** 3 torso and 7 arm joints (rad), finger and thumb closure. **Action:** 12 joint and closure targets.
- **Split used for training** (`configs/act_split_v2.json` in the repository): 360 training and 40 validation
  episodes. The 20 v1 validation episodes stay in validation.

**Known video issue:** 11 single frames in the wrist-camera video have an AV1 encoding glitch, a rectangle in the top
right that shows grey instead of the table. The raw recorded frames are clean, so it is the encoding, not the
simulation. None of these episodes is in validation. Seeds, with episode and frame:
1220 (216, frame 581), 1230 (225, 615), 1244 (239, 571), 1290 (283, 621), 1304 (297, 575), 1328 (321, 577),
1332 (325, 629), 1346 (338, 583), 1404 (394, 619 and 621), 1406 (396, 581).
Also a smaller patch in the eye camera: seed 1314 (episode 307, frame 681). That is 11 of 196,596 wrist frames;
the scan covered episodes 200-399 only. Details: `docs/eval/v2_video_scan.txt` in the repository.
