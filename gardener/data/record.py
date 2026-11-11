"""Record an expert episode as raw arrays: camera images, joint state, action, instruction.

Runs in the `gardener` env, next to the simulation. scripts/to_lerobot.py then converts these files in
the `gardener-train` env, because LeRobot needs other versions of torch and numpy (book section 8, Phase 0).

One frame holds:
  state:  measured joint angles (3 torso, 7 arm, radians), then the commanded finger and thumb closure (0 to 1)
  action: the joint targets and closures commanded for this frame, in the same order
  images: the eye camera, a wrist camera and a fixed scene camera, 224 x 224 RGB

The robot is drawn with iCub's real meshes, on a second copy loaded only for drawing. The copy has no physics
and takes every joint angle from the simulated robot, so what the cameras show cannot change an episode. The
plants are drawn only too: water passes through them and they stay below the lowest the can comes.
"""
from __future__ import annotations

import math

import numpy as np
import threepp as tp

from gardener.expert.planner import PLAN_JOINTS

FPS = 25                  # the simulation runs at 50 steps/s
STEPS_PER_SECOND = 50
IMAGE_SIZE = 224
SCENE_CAMERA = ((-0.95, 0.08, 1.15), (-0.32, 0.08, 0.62))   # position and look-at point
EYE_LOOK_AT = (-0.34, 0.08, 0.60)
# The wrist camera rides on the hand, above the fingers, like a small mast. Of six placements compared in one
# episode, it alone keeps the fingers, handle, can, target pot and water stream in view through every phase.
# It has no collider. The forearm blocks the eye camera during a pour, but it cannot come between these.
EYE_NEAR = 0.10           # m
WRIST_CAMERA = ((-0.12, -0.02, 0.12), (-0.06, -0.20, -0.05))   # position and look-at point, hand frame (m)
WATER_DRAW_RADIUS = 0.004
PLANT_OFFSET = (0.024, 0.0)   # off the pot's centre, so the water stays in view
COLOURS = dict(stem=0x4c8c32, leaf=0x7ccf4f, robot=0x9aa4b2, hand=0xd8c0a0, can=0xc0392b, table=0x8a6a48, pot=0x4f7942, floor=0x2b2f36, water=0x2f7fe0)


class EpisodeRecorder:
    """Pass to run_episode(recorder=...). It only reads the simulation, so it cannot change an episode."""

    def __init__(self, fps: int = FPS, real_look: bool = True):
        """fps: frames kept per second. At a rate that does not divide 50, frames sit up to 10 ms from even spacing.
        real_look: draw iCub's real meshes; False draws its collision shapes."""
        self.fps = fps
        self.real_look = real_look
        self.physics_robot = None
        self.visual = None
        self.canvas = tp.Canvas("recorder", width=IMAGE_SIZE, height=IMAGE_SIZE, antialiasing=0, vsync=False, headless=True)
        self.renderer = tp.GLRenderer(self.canvas)
        self.renderer.set_clear_color(0xcfd8e3)
        self.scene = tp.Scene()
        self.scene.add(tp.HemisphereLight(0xffffff, 0x50545c, 1.1))
        sun = tp.DirectionalLight(0xffffff, 1.6)
        sun.position.set(-1.0, 0.6, 3.0)
        self.scene.add(sun)
        self.scene_cam = self._camera(*SCENE_CAMERA)
        # The camera sits at the eye link, inside the head's solid shell: it starts drawing beyond the skull.
        self.eye_cam = tp.PerspectiveCamera(60, 1.0, EYE_NEAR, 5.0)
        self.eye_cam.up.set(0, 0, 1)
        self.wrist_cam = tp.PerspectiveCamera(70, 1.0, 0.01, 5.0)
        self.water = None
        self.matrix = tp.Matrix4()
        self.steps = 0
        self.state, self.action, self.eye, self.top, self.wrist = [], [], [], [], []

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

    def add_robot(self, robot, hand_parts, base_position=(0.0, 0.0, 0.05)) -> None:
        if self.real_look:
            self.physics_robot = robot
            self.visual = tp.URDFLoader().load(robot.urdf_path)
            self.visual.position.set(*base_position)
            self._hide_wireframes(self.visual)
            names = [j.name for j in self.visual.get_articulated_joint_info()]
            self.moving = [(names.index(n), robot.index[n]) for n in names if n in robot.index and robot.upper[robot.index[n]] - robot.lower[robot.index[n]] > 1e-3]
            self.scene.add(self.visual)
            return
        for name in robot.body_names:
            self._show(robot.collider(name), COLOURS["hand"] if name in hand_parts else COLOURS["robot"])

    def _pose_robot(self) -> None:
        """Copy the simulated robot's joint angles to the copy that is drawn."""
        if self.visual is not None:
            angles = self.physics_robot.positions()
            for drawn, simulated in self.moving:
                self.visual.set_joint_value(drawn, float(angles[simulated]))

    @classmethod
    def _hide_wireframes(cls, obj) -> None:
        """The loader adds a wireframe of each collision shape over the real meshes. The model's red parts
        become grey, as on the real robot and so that the red can stands out from the arm."""
        material = getattr(obj, "material", None)
        if getattr(material, "wireframe", False):
            obj.visible = False
        elif material is not None and material.color.r > 1.5 * material.color.g:
            material.color = COLOURS["robot"]
        for child in obj.children:
            cls._hide_wireframes(child)

    def _add_plant(self, pot) -> None:
        """A small sprout: a stem and five leaves, all inside the pot."""
        def blob(colour, position, size):
            material = tp.MeshStandardMaterial()
            material.color = colour
            mesh = tp.Mesh(tp.SphereGeometry(1.0, 12, 8), material)
            mesh.position.set(*position)
            mesh.scale.set(*size)
            self.scene.add(mesh)

        x, y = pot.x + PLANT_OFFSET[0], pot.y + PLANT_OFFSET[1]
        rim = pot.z + pot.height
        soil = pot.z + pot.floor
        blob(COLOURS["stem"], (x, y, (soil + rim) / 2), (0.003, 0.003, (rim - soil) / 2))
        for k in range(5):
            a = 2 * math.pi * k / 5
            blob(COLOURS["leaf"], (x + 0.011 * math.cos(a), y + 0.011 * math.sin(a), rim - 0.006 + 0.002 * (k % 2)), (0.011, 0.006, 0.004))

    def add_scenery(self, table_mesh, pot_meshes, can_meshes, n_particles: int, table_z: float, pots=()) -> None:
        for pot in pots:
            self._add_plant(pot)
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

    def frame(self, eye_position, T_hand: np.ndarray, q: dict, fingers: float, thumb: float, targets: dict, particles: np.ndarray) -> None:
        """Called once per simulation step; keeps every EVERY_N_STEPS-th."""
        self.steps += 1
        if self.steps * self.fps // STEPS_PER_SECOND == (self.steps - 1) * self.fps // STEPS_PER_SECOND:
            return
        self._pose_robot()
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
        position = T_hand[:3, :3] @ np.array(WRIST_CAMERA[0]) + T_hand[:3, 3]
        target = T_hand[:3, :3] @ np.array(WRIST_CAMERA[1]) + T_hand[:3, 3]
        self.wrist_cam.up.set(*map(float, T_hand[:3, 2]))     # the hand's z axis, so the view turns with the hand
        self.wrist_cam.position.set(*map(float, position))
        self.wrist_cam.look_at(*map(float, target))
        self.renderer.render(self.scene, self.wrist_cam)
        self.wrist.append(self.renderer.read_pixels()[..., :3].copy())
        self.state.append([q[n] for n in PLAN_JOINTS] + [fingers, thumb])
        self.action.append([targets[n] for n in PLAN_JOINTS] + [fingers, thumb])

    def save(self, path: str, instruction: str, seed: int, metadata: dict) -> None:
        np.savez_compressed(path, eye=np.stack(self.eye), scene=np.stack(self.top), wrist=np.stack(self.wrist),
                 state=np.asarray(self.state, dtype=np.float32), action=np.asarray(self.action, dtype=np.float32),
                 instruction=instruction, seed=seed, fps=self.fps, joints=np.array(PLAN_JOINTS + ["fingers", "thumb"]),
                 **{f"meta_{k}": v for k, v in metadata.items()})
