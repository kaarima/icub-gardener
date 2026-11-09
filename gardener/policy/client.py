"""Controllers for run_episode(controller=...): a trained policy, or the actions saved in an episode.

The policy runs in the gardener-train env (LeRobot and its torch), the simulation in gardener, so
PolicyController starts scripts/policy_server.py there and passes observations over a pipe.
"""
from __future__ import annotations

import os
import pickle
import struct
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAIN_PYTHON = os.path.expanduser("~/miniconda3/envs/gardener-train/bin/python")


class PolicyController:
    def __init__(self, checkpoint: str, n_action_steps: int | None = None, ensemble: float | None = None,
                 device: str | None = None):
        cmd = [TRAIN_PYTHON, os.path.join(ROOT, "scripts", "policy_server.py"), "--checkpoint", checkpoint]
        if n_action_steps is not None:
            cmd += ["--n-action-steps", str(n_action_steps)]
        if ensemble is not None:
            cmd += ["--ensemble", str(ensemble)]
        if device is not None:
            cmd += ["--device", device]
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}     # the ROS PYTHONPATH breaks LeRobot
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
        self._ask({"reset": True})

    def _ask(self, msg: dict) -> dict:
        data = pickle.dumps(msg)
        self.proc.stdin.write(struct.pack("<I", len(data)) + data)
        self.proc.stdin.flush()
        head = self.proc.stdout.read(4)
        if len(head) < 4:
            raise RuntimeError(f"policy server stopped (exit code {self.proc.poll()}); its error is printed above")
        return pickle.loads(self.proc.stdout.read(struct.unpack("<I", head)[0]))

    def __call__(self, obs: dict) -> np.ndarray:
        return self._ask({"obs": obs})["action"]

    def close(self):
        self.proc.stdin.close()
        self.proc.wait(timeout=30)


class ReplayController:
    """Plays an episode's saved actions back, open loop, one per frame, and keeps the states it sees."""

    def __init__(self, episode_path: str):
        e = np.load(episode_path)
        self.actions = e["action"]
        self.saved_states = e["state"]
        self.seen = []

    def __call__(self, obs: dict) -> np.ndarray:
        k = len(self.seen)
        self.seen.append(obs["state"])
        return self.actions[min(k, len(self.actions) - 1)]

    def deviation_deg(self) -> tuple[float, float, int]:
        """Largest and median joint difference between replayed and saved states, over the frames both have."""
        n = min(len(self.seen), len(self.saved_states))
        d = np.degrees(np.abs(np.asarray(self.seen[:n])[:, :10] - self.saved_states[:n, :10]))
        return float(d.max()), float(np.median(d.max(axis=1))), n

    def close(self):
        pass
