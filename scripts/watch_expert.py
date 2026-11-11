#!/usr/bin/env python3
"""Watch one expert episode live in a window. Drag to orbit, scroll to zoom. Saves nothing.

The robot is drawn from its collision shapes, as in the recorded images, so it looks like boxes and capsules.
The window appears after about 5 s of planning and settling, when the can and table are placed.

Usage (conda env gardener):
  python scripts/watch_expert.py --seed 1 --instruction "Water the right plant."
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import threepp as tp  # noqa: E402

from gardener.data.record import COLOURS, SCENE_CAMERA, EpisodeRecorder  # noqa: E402
from gardener.expert.state_machine import run_episode  # noqa: E402

DEFAULT_URDF = "~/icub-assets/iCub/robots/iCubGazeboV2_10/gardener_hand.urdf"


class LiveViewer(EpisodeRecorder):
    """Same drawing as the recorder, in a window, keeping nothing."""

    def __init__(self, headless=False, stop_after=None):
        self.canvas = tp.Canvas("iCub gardener: live episode", width=1000, height=800, antialiasing=4, vsync=False, headless=headless)
        self.renderer = tp.GLRenderer(self.canvas)
        self.renderer.set_clear_color(0xcfd8e3)
        self.scene = tp.Scene()
        self.scene.add(tp.HemisphereLight(0xffffff, 0x50545c, 1.1))
        sun = tp.DirectionalLight(0xffffff, 1.6)
        sun.position.set(-1.0, 0.6, 3.0)
        self.scene.add(sun)
        floor = tp.Mesh(tp.PlaneGeometry(20, 20), tp.MeshStandardMaterial())
        floor.material.color = COLOURS["floor"]
        self.scene.add(floor)
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

    def frame(self, eye_position, T_hand, q, fingers, thumb, targets, particles):
        self.frames += 1
        if self.frames % 2:
            return
        self._pose_robot()
        for i, p in enumerate(particles):
            self.matrix.set_position(float(p[0]), float(p[1]), float(p[2]))
            self.water.set_matrix_at(i, self.matrix)
        self.water.instance_matrix_needs_update()

        def draw():
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
    ap.add_argument("--test-frames", type=int, default=None, help="no window: run this many steps, then stop (to test this script)")
    args = ap.parse_args()
    viewer = LiveViewer(headless=args.test_frames is not None, stop_after=args.test_frames)
    r = run_episode(os.path.expanduser(args.urdf), os.path.join(ROOT, "configs", "scene.yaml"), os.path.join(ROOT, "configs", "poses.yaml"),
                    args.instruction, args.seed, recorder=viewer)
    print(f"RESULT: {'SUCCESS' if r.success else 'FAILED: ' + r.failure}; poured {r.poured_fraction * 100:.0f}% of the can, "
          f"{r.water_in_target * 100:.0f}% of that in the target pot; can placed {r.placed_error_mm:.0f} mm from its start")


if __name__ == "__main__":
    main()
