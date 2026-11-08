#!/usr/bin/env python3
"""Watch iCub hold still and play a short arm + hand sequence.

    python scripts/view_icub.py                  # window (drag to orbit)
    python scripts/view_icub.py --test           # no window: physics only, prints a stability report
    python scripts/view_icub.py --colliders      # also draw the collision shapes (yellow wireframe)

Poses and the sequence live in configs/poses.yaml (degrees).
Every second it prints the phase and two numbers for the 30 controlled joints:
  err   = largest gap between target and actual angle (deg)
  speed = largest joint speed (deg/s); during a "hold" it should be near 0
"""
import argparse
import math
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gardener.sim.robot import ICub, TORSO, RIGHT_ARM, make_world  # noqa: E402

DEFAULT_URDF = "~/icub-assets/iCub/robots/iCubGazeboV2_10/gardener_hand.urdf"
DT = 0.02  # seconds of simulation per frame
SETTLE = 0.5  # seconds after a move before a hold counts as "settled" in the report


class Sequence:
    """Turns the YAML steps into targets at any time t (smooth, looping)."""

    def __init__(self, cfg: dict):
        poses = {name: {j: math.radians(v) for j, v in p.items()} for name, p in cfg["poses"].items()}
        joints = TORSO + RIGHT_ARM
        self.steps, t0 = [], 0.0
        prev_pose, prev_hand = poses[cfg["sequence"][0][0]], cfg["sequence"][0][1]
        for pose_name, hand, move, hold in cfg["sequence"]:
            pose = poses[pose_name]
            self.steps.append(dict(
                name=f"{pose_name}, hand={hand:.1f}", t0=t0, move=float(move), hold=float(hold),
                a0={j: prev_pose.get(j, 0.0) for j in joints},
                a1={j: pose.get(j, 0.0) for j in joints},
                h0=float(prev_hand), h1=float(hand)))
            t0 += move + hold
            prev_pose, prev_hand = pose, hand
        self.duration = t0

    def sample(self, t: float):
        t = t % self.duration
        for s in self.steps:
            if t < s["t0"] + s["move"] + s["hold"]:
                u = min(1.0, (t - s["t0"]) / max(s["move"], 1e-6))
                u = u * u * (3 - 2 * u)  # smoothstep: gentle start and stop
                arm = {j: s["a0"][j] + (s["a1"][j] - s["a0"][j]) * u for j in s["a0"]}
                hand = s["h0"] + (s["h1"] - s["h0"]) * u
                holding = (t - s["t0"]) >= s["move"]
                settled = (t - s["t0"]) >= s["move"] + SETTLE
                return arm, hand, s["name"], holding, settled
        raise RuntimeError("unreachable")


def build_scene(robot, show_colliders=False):
    import threepp as tp
    canvas = tp.Canvas("iCub gardener", width=900, height=900, antialiasing=4)
    rend = tp.GLRenderer(canvas)
    rend.set_clear_color(0x0e1118)
    rend.shadow_map_enabled = True
    rend.tone_mapping = tp.ToneMapping.ACESFilmic
    scene = tp.Scene()
    scene.add(tp.HemisphereLight(0xbcd0ff, 0x20242c, 0.95))
    key = tp.DirectionalLight(0xffffff, 2.6)
    key.position.set(3, -3, 7)
    key.cast_shadow = True
    scene.add(key)
    floor = tp.Mesh(tp.PlaneGeometry(20, 20), tp.MeshStandardMaterial())
    floor.material.color = 0x171b24
    floor.material.roughness = 0.95
    floor.receive_shadow = True
    scene.add(floor)
    for m in robot.meshes:
        m.cast_shadow = True
        if show_colliders:
            # robot.meshes are the (normally hidden) collision shapes; their
            # children are the pretty visual meshes. Show the shapes as
            # yellow wireframes: this is what the physics actually touches.
            m.material.visible = True
            m.material.wireframe = True
            m.material.color = 0xffd23f
        scene.add(m)
    camera = tp.PerspectiveCamera(45, 1.0, 0.01, 100)
    camera.up.set(0, 0, 1)                 # the world is z-up
    # iCub faces -x (its root frame's x axis points backwards) and its right
    # side is +y, so this camera sits in front of him, on his right.
    camera.position.set(-1.5, 1.1, 1.2)
    camera.look_at(0, 0, 0.8)
    controls = tp.OrbitControls(camera, canvas)
    controls.enable_damping = True
    controls.target = tp.Vector3(0, 0, 0.8)

    def on_resize(w, h):
        camera.aspect = w / max(h, 1)
        camera.update_projection_matrix()
        rend.set_size(w, h)
    canvas.on_window_resize(on_resize)
    return canvas, rend, scene, camera, controls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", default=DEFAULT_URDF)
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "configs", "poses.yaml"))
    ap.add_argument("--test", action="store_true", help="no window: run one loop of the sequence and report")
    ap.add_argument("--colliders", action="store_true", help="draw the collision shapes as yellow wireframes")
    args = ap.parse_args()

    with open(args.config) as f:
        seq = Sequence(yaml.safe_load(f))

    world = make_world()
    robot = ICub(world, args.urdf)
    print(f"loaded {robot.num_dof} joints; sequence lasts {seq.duration:.1f} s")

    state = {"t": 0.0, "frame": 0, "hold_err": 0.0, "hold_speed": 0.0}

    def step():
        arm, hand, name, holding, settled = seq.sample(state["t"])
        robot.set_targets(arm)
        robot.set_targets(robot.hand_targets(hand))
        robot.apply()
        world.step(DT)
        state["t"] += DT
        state["frame"] += 1
        err, speed = math.degrees(robot.tracking_error()), math.degrees(robot.max_speed())
        if settled:
            state["hold_err"] = max(state["hold_err"], err)
            state["hold_speed"] = max(state["hold_speed"], speed)
        if state["frame"] % int(1 / DT) == 0:
            tag = "hold" if holding else "move"
            print(f"t={state['t']:5.1f}s  {tag}  {name:<26} err={err:6.2f} deg  speed={speed:7.2f} deg/s")

    if args.test:
        while state["t"] < seq.duration:
            step()
        print(f"\nsettled holds (>{SETTLE}s after each move): worst err = {state['hold_err']:.2f} deg, worst speed = {state['hold_speed']:.2f} deg/s")
        return

    canvas, rend, scene, camera, controls = build_scene(robot, show_colliders=args.colliders)

    def frame():
        step()
        controls.update()
        rend.render(scene, camera)

    canvas.animate(frame)


if __name__ == "__main__":
    main()
