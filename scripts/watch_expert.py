#!/usr/bin/env python3
"""Watch one episode live in a window: the expert, or a trained policy with --policy. Drag to orbit, scroll
to zoom. Saves nothing.

iCub is drawn with its real meshes, in grey, on a physics-free copy that follows the simulated joints, as in
the recorded images (gardener/data/record.py). The physics itself uses the simplified collision shapes; use
scripts/view_icub.py --colliders to see those. The window adds a dark floor that the recorded images do not have.
The window appears after about 5 s of planning and settling, when the can and table are placed.

With --policy, a trained checkpoint drives the robot instead of the expert, after the same setup, and is
scored the same way (run_controller in gardener/expert/state_machine.py). The policy needs the three
224 x 224 camera images every frame: they are drawn in a corner of this window and read back before the
main view is drawn over them, with the recorder's own cameras, so they match the recorded episodes. The
window then has no antialiasing, as the recorder has none. Each policy frame waits for the policy server,
so the episode may run slower than real time.

Usage (conda env gardener):
  python scripts/watch_expert.py --seed 1 --instruction "Water the right plant."
  python scripts/watch_expert.py --seed 5038 --instruction "Give the left plant just a little water." \
      --policy outputs/act/checkpoints/080000/pretrained_model
"""
import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import threepp as tp  # noqa: E402

from gardener.data.record import COLOURS, EYE_NEAR, IMAGE_SIZE, SCENE_CAMERA, EpisodeRecorder  # noqa: E402
from gardener.expert.state_machine import PLAN_JOINTS, run_episode  # noqa: E402

DEFAULT_URDF = "~/icub-assets/iCub/robots/iCubGazeboV2_10/gardener_hand.urdf"


class LiveViewer(EpisodeRecorder):
    """Same drawing as the recorder, in a window. Keeps nothing, unless capture is on: then it also makes the
    recorder's three camera images each frame, for a policy (run_controller takes and clears them)."""

    def __init__(self, headless=False, stop_after=None, capture=False):
        self.canvas = tp.Canvas("iCub gardener: live episode", width=1000, height=800, antialiasing=0 if capture else 4,
                                vsync=False, headless=headless)
        self.renderer = tp.GLRenderer(self.canvas)
        self.renderer.set_clear_color(0xcfd8e3)
        self.scene = tp.Scene()
        self.scene.add(tp.HemisphereLight(0xffffff, 0x50545c, 1.1))
        sun = tp.DirectionalLight(0xffffff, 1.6)
        sun.position.set(-1.0, 0.6, 3.0)
        self.scene.add(sun)
        self.floor = tp.Mesh(tp.PlaneGeometry(20, 20), tp.MeshStandardMaterial())   # the window's only: the recorder has no floor
        self.floor.material.color = COLOURS["floor"]
        self.scene.add(self.floor)
        self.camera = tp.PerspectiveCamera(50, 1000 / 800, 0.02, 30.0)
        self.camera.up.set(0, 0, 1)
        self.camera.position.set(*SCENE_CAMERA[0])
        self.camera.look_at(*SCENE_CAMERA[1])
        self.controls = tp.OrbitControls(self.camera, self.canvas)
        self.controls.target = tp.Vector3(*SCENE_CAMERA[1])
        self.water = None
        self.real_look = True
        self.physics_robot = None
        self.visual = None
        self.matrix = tp.Matrix4()
        self.frames = 0
        self.stop_after = stop_after
        self.started = time.perf_counter()
        self.capture = capture
        self.state, self.action, self.eye, self.top, self.wrist = [], [], [], [], []
        if capture:   # the recorder's cameras, as in EpisodeRecorder.__init__
            self.scene_cam = self._camera(*SCENE_CAMERA)
            self.eye_cam = tp.PerspectiveCamera(60, 1.0, EYE_NEAR, 5.0)
            self.eye_cam.up.set(0, 0, 1)
            self.wrist_cam = tp.PerspectiveCamera(70, 1.0, 0.01, 5.0)

    def frame(self, eye_position, T_hand, q, fingers, thumb, targets, particles):
        self.frames += 1
        if self.frames % 2:     # 50 steps/s; draw every second one, the same frames the recorder keeps at 25 fps
            return
        self._pose_robot()
        for i, p in enumerate(particles):
            self.matrix.set_position(float(p[0]), float(p[1]), float(p[2]))
            self.water.set_matrix_at(i, self.matrix)
        self.water.instance_matrix_needs_update()
        if self.capture:
            self.floor.visible = False
            for kept, cam in self._place_cameras(eye_position, T_hand):
                self.renderer.set_viewport(0, 0, IMAGE_SIZE, IMAGE_SIZE)
                self.renderer.render(self.scene, cam)
                kept.append(self.renderer.read_pixels()[-IMAGE_SIZE:, :IMAGE_SIZE, :3].copy())   # bottom-left corner
            self.floor.visible = True
            self.state.append([q[n] for n in PLAN_JOINTS] + [fingers, thumb])
            self.action.append([targets[n] for n in PLAN_JOINTS] + [fingers, thumb])

        def draw():
            self.renderer.set_viewport(0, 0, *self.canvas.size())
            self.controls.update()
            self.renderer.render(self.scene, self.camera)

        open_ = self.canvas.animate_once(draw)
        # Keep to real time when the simulation runs faster than that.
        wait = self.frames * 0.02 - (time.perf_counter() - self.started)
        if wait > 0:
            time.sleep(wait)
        if not open_ or (self.stop_after and self.frames >= self.stop_after):
            print(f"stopped after {self.frames * 0.02:.1f} s of the episode")
            raise SystemExit(0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--urdf", default=DEFAULT_URDF)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--instruction", default="Water the right plant.")
    ap.add_argument("--policy", default=None, help="watch this trained policy (a pretrained_model folder) instead of the expert")
    ap.add_argument("--n-action-steps", type=int, default=None, help="with --policy: actions run per predicted chunk")
    ap.add_argument("--ensemble", type=float, default=None, help="with --policy: temporal ensembling coefficient")
    ap.add_argument("--test-frames", type=int, default=None, help="no window: run this many steps, then stop (to test this script)")
    ap.add_argument("--test-images", default=None,
                    help="with --test-frames: also make the policy's camera images and save them to this .npz (to compare with a recording)")
    args = ap.parse_args()
    controller = None
    if args.policy:
        from gardener.policy.client import PolicyController
        controller = PolicyController(args.policy, n_action_steps=args.n_action_steps, ensemble=args.ensemble)
    viewer = LiveViewer(headless=args.test_frames is not None, stop_after=args.test_frames,
                        capture=controller is not None or args.test_images is not None)
    try:
        r = run_episode(os.path.expanduser(args.urdf), os.path.join(ROOT, "configs", "scene.yaml"), os.path.join(ROOT, "configs", "poses.yaml"),
                        args.instruction, args.seed, recorder=viewer, controller=controller)
    except SystemExit:
        if args.test_images:
            np.savez_compressed(args.test_images, eye=np.stack(viewer.eye), scene=np.stack(viewer.top), wrist=np.stack(viewer.wrist))
            print(f"saved {len(viewer.eye)} frames of camera images to {args.test_images}")
        raise
    finally:
        if controller is not None:
            controller.close()
    print(f"RESULT: {'SUCCESS' if r.success else 'FAILED: ' + r.failure}; poured {r.poured_fraction * 100:.0f}% of the can, "
          f"{r.water_in_target * 100:.0f}% of that in the target pot; can placed {r.placed_error_mm:.0f} mm from its start")
    if not r.success:
        print("failure reasons: " + "; ".join(r.failures or [r.failure]))


if __name__ == "__main__":
    main()
