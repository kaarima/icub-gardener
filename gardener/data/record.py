"""Record an expert episode as raw arrays: camera images, joint state, action, instruction.

Runs in the `gardener` env, next to the simulation. scripts/to_lerobot.py then converts these files in
the `gardener-train` env, because LeRobot needs other versions of torch and numpy (book section 8, Phase 0).

One frame holds:
  state:  measured joint angles (3 torso, 7 arm, radians), then the commanded finger and thumb closure (0 to 1)
  action: the joint targets and closures commanded for this frame, in the same order
  images: the eye camera and a fixed scene camera, 224 x 224 RGB

Limitation: the robot is drawn from its collision shapes, because the build with separate finger drives has
no visual meshes. A policy trained on these images sees capsules and boxes, not iCub's real look.
"""
from __future__ import annotations

import numpy as np
import threepp as tp

from gardener.expert.planner import PLAN_JOINTS

FPS = 25                  # the simulation runs at 50 steps/s, so one frame every 2 steps
EVERY_N_STEPS = 2
IMAGE_SIZE = 224
SCENE_CAMERA = ((-0.95, 0.08, 1.15), (-0.32, 0.08, 0.62))   # position and look-at point
EYE_LOOK_AT = (-0.34, 0.08, 0.60)
WATER_DRAW_RADIUS = 0.004
COLOURS = dict(robot=0x9aa4b2, hand=0xd8c0a0, can=0xc0392b, table=0x8a6a48, pot=0x4f7942, floor=0x2b2f36, water=0x2f7fe0)


class EpisodeRecorder:
    """Pass to run_episode(recorder=...). It only reads the simulation, so it cannot change an episode."""

    def __init__(self):
        self.canvas = tp.Canvas("recorder", width=IMAGE_SIZE, height=IMAGE_SIZE, antialiasing=0, vsync=False, headless=True)
        self.renderer = tp.GLRenderer(self.canvas)
        self.renderer.set_clear_color(0xcfd8e3)
        self.scene = tp.Scene()
        self.scene.add(tp.HemisphereLight(0xffffff, 0x50545c, 1.1))
        sun = tp.DirectionalLight(0xffffff, 1.6)
        sun.position.set(-1.0, 0.6, 3.0)
        self.scene.add(sun)
        self.scene_cam = self._camera(*SCENE_CAMERA)
        self.eye_cam = tp.PerspectiveCamera(60, 1.0, 0.02, 5.0)
        self.eye_cam.up.set(0, 0, 1)
        self.water = None
        self.matrix = tp.Matrix4()
        self.steps = 0
        self.state, self.action, self.eye, self.top = [], [], [], []

    @staticmethod
    def _camera(position, look_at):
        cam = tp.PerspectiveCamera(50, 1.0, 0.02, 6.0)
        cam.up.set(0, 0, 1)
        cam.position.set(*position)
        cam.look_at(*look_at)
        return cam

    def _show(self, mesh, colour):
        mesh.material.visible = True
        mesh.material.color = colour
        mesh.material.roughness = 0.8
        self.scene.add(mesh)

    def add_robot(self, robot, hand_parts) -> None:
        for name in robot.body_names:
            self._show(robot.collider(name), COLOURS["hand"] if name in hand_parts else COLOURS["robot"])

    def add_scenery(self, table_mesh, pot_meshes, can_meshes, n_particles: int, table_z: float) -> None:
        self._show(table_mesh, COLOURS["table"])
        for mesh in pot_meshes:
            self._show(mesh, COLOURS["pot"])
        for mesh in can_meshes:
            self._show(mesh, COLOURS["can"])
        material = tp.MeshStandardMaterial()
        material.color = COLOURS["water"]
        self.water = tp.InstancedMesh(tp.SphereGeometry(WATER_DRAW_RADIUS, 6, 4), material, n_particles)
        self.water.frustum_culled = False
        self.scene.add(self.water)

    def frame(self, eye_position, q: dict, fingers: float, thumb: float, targets: dict, particles: np.ndarray) -> None:
        """Called once per simulation step; keeps every EVERY_N_STEPS-th."""
        self.steps += 1
        if self.steps % EVERY_N_STEPS:
            return
        for i, p in enumerate(particles):
            self.matrix.set_position(float(p[0]), float(p[1]), float(p[2]))
            self.water.set_matrix_at(i, self.matrix)
        self.water.instance_matrix_needs_update()
        self.eye_cam.position.set(*map(float, eye_position))
        self.eye_cam.look_at(*EYE_LOOK_AT)
        self.renderer.render(self.scene, self.eye_cam)
        self.eye.append(self.renderer.read_pixels()[..., :3].copy())
        self.renderer.render(self.scene, self.scene_cam)
        self.top.append(self.renderer.read_pixels()[..., :3].copy())
        self.state.append([q[n] for n in PLAN_JOINTS] + [fingers, thumb])
        self.action.append([targets[n] for n in PLAN_JOINTS] + [fingers, thumb])

    def save(self, path: str, instruction: str, seed: int, metadata: dict) -> None:
        np.savez(path, eye=np.stack(self.eye), scene=np.stack(self.top),
                 state=np.asarray(self.state, dtype=np.float32), action=np.asarray(self.action, dtype=np.float32),
                 instruction=instruction, seed=seed, fps=FPS, joints=np.array(PLAN_JOINTS + ["fingers", "thumb"]),
                 **{f"meta_{k}": v for k, v in metadata.items()})
