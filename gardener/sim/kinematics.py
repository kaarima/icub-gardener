"""Forward and inverse kinematics of iCub's right arm, from the URDF. No physics.

All poses are 4x4 matrices in the world frame, metres and radians. Joint angles are passed as a dict
{joint name: angle}; joints left out stay at 0.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import numpy as np

from gardener.sim.builder import load_shapes


def rpy_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def quat_matrix(w, x, y, z):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _rotation(axis, angle):
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


class ArmKinematics:
    """Chains from the base to any link, a position and orientation IK, and table clearance."""

    ARM_LINKS = ["chest", "r_upper_arm", "r_elbow_1", "r_forearm", "r_wrist_1", "r_hand"]

    def __init__(self, urdf: str, base=(0.0, 0.0, 0.0)):
        self.urdf = urdf
        self.root = ET.parse(urdf).getroot()
        self.base = np.array(base, dtype=float)
        self._joints = {j.find("child").get("link"): j for j in self.root.findall("joint")}
        self._by_name = {j.get("name"): j for j in self.root.findall("joint")}
        self._chains = {}
        self._link_points = None

    def _chain(self, link):
        if link not in self._chains:
            out, name = [], link
            while name in self._joints:
                j = self._joints[name]
                o = j.find("origin")
                T = np.eye(4)
                if o is not None:
                    T[:3, :3] = rpy_matrix(*[float(v) for v in o.get("rpy", "0 0 0").split()])
                    T[:3, 3] = [float(v) for v in o.get("xyz", "0 0 0").split()]
                axis = None
                if j.get("type") != "fixed":
                    axis = np.array([float(v) for v in j.find("axis").get("xyz").split()])
                    axis /= np.linalg.norm(axis)
                out.append((j.get("name"), T, axis))
                name = j.find("parent").get("link")
            self._chains[link] = out[::-1]
        return self._chains[link]

    def limits(self, name):
        lim = self._by_name[name].find("limit")
        return float(lim.get("lower")), float(lim.get("upper"))

    def fk(self, link, q):
        T = np.eye(4)
        T[:3, 3] = self.base
        for name, origin, axis in self._chain(link):
            T = T @ origin
            if axis is not None and name in q:
                M = np.eye(4)
                M[:3, :3] = _rotation(axis, q[name])
                T = T @ M
        return T

    def hand(self, q):
        return self.fk("r_hand", q)

    def jacobian(self, q, names):
        """6 x len(names) Jacobian of the r_hand frame, and that frame's pose."""
        T = np.eye(4)
        T[:3, 3] = self.base
        axes = {}
        for name, origin, axis in self._chain("r_hand"):
            T = T @ origin
            if axis is not None:
                axes[name] = (T[:3, :3] @ axis, T[:3, 3].copy())
                if name in q:
                    M = np.eye(4)
                    M[:3, :3] = _rotation(axis, q[name])
                    T = T @ M
        J = np.zeros((6, len(names)))
        for i, n in enumerate(names):
            a, p = axes[n]
            J[:3, i] = np.cross(a, T[:3, 3] - p)
            J[3:, i] = a
        return J, T

    def ik(self, q0, T_goal, names, center_weights=None, iterations=80, damping=1e-4, rest=None):
        """Damped least squares towards T_goal. Spare freedom pulls weighted joints to a rest angle.

        center_weights: {joint name: weight}. rest: {joint name: angle}; joints left out rest at
        mid-range. Returns (q, position error m, rotation error rad).
        """
        q = dict(q0)
        lo = np.array([self.limits(n)[0] for n in names])
        hi = np.array([self.limits(n)[1] for n in names])
        mid, span = (lo + hi) / 2, np.maximum(hi - lo, 1e-6)
        if rest:
            mid = np.array([rest.get(n, m) for n, m in zip(names, mid)])
        w = np.array([(center_weights or {}).get(n, 0.0) for n in names])
        for _ in range(iterations):
            J, T = self.jacobian(q, names)
            dR = T_goal[:3, :3] @ T[:3, :3].T
            e = np.concatenate([T_goal[:3, 3] - T[:3, 3],
                                0.5 * np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])])
            x = np.array([q[n] for n in names])
            JJt = J @ J.T + damping * np.eye(6)
            step = J.T @ np.linalg.solve(JJt, e)
            N = np.eye(len(names)) - J.T @ np.linalg.solve(JJt, J)
            step += N @ (-0.2 * w * (x - mid) / span)
            x = np.clip(x + np.clip(step, -0.2, 0.2), lo, hi)
            q.update(dict(zip(names, x.tolist())))
        T = self.hand(q)
        dR = T_goal[:3, :3] @ T[:3, :3].T
        angle = math.acos(float(np.clip((np.trace(dR) - 1) / 2, -1.0, 1.0)))
        return q, float(np.linalg.norm(T_goal[:3, 3] - T[:3, 3])), angle

    def solve_lift(self, q0, names, lower, upper, dz):
        """Joint angles that raise the hand by dz, keeping its orientation. Returns (q, error m)."""
        T = self.hand(q0)
        T[2, 3] += dz
        q, p_err, r_err = self.ik(q0, T, names, iterations=40)
        return q, p_err + r_err * 0.1

    # ---- clearance -------------------------------------------------------
    def _points(self):
        """Sample points of each arm collider in its link frame, with a radius (0 for box corners)."""
        if self._link_points is None:
            shapes = {e["name"]: e for e in load_shapes(self.urdf)}
            zero = {}
            base = self.base.copy()
            self.base = np.zeros(3)
            for name in self.ARM_LINKS:
                zero[name] = self.fk(name, {})
            self.base = base
            self._link_points = {}
            for name in self.ARM_LINKS:
                e = shapes[name]
                R = quat_matrix(*e["quat"])
                c = np.array(e["pos"])
                if e["kind"] == "capsule":
                    # threepp builds a capsule along the mesh's local y axis
                    half = R[:, 1] * e["length"] / 2
                    pts, radius = np.linspace(c - half, c + half, 9), e["radius"]
                else:
                    s = np.array(e["size"]) / 2
                    pts = np.array([c + R @ (s * np.array([i, j, k])) for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)])
                    radius = 0.0
                inv = np.linalg.inv(zero[name])
                self._link_points[name] = ((inv[:3, :3] @ pts.T).T + inv[:3, 3], radius)
        return self._link_points

    def table_clearance(self, q, table_z, table_back_x):
        """Lowest point of each arm collider above the table (m); only points over the table count."""
        out = {}
        for name, (pts, radius) in self._points().items():
            T = self.fk(name, q)
            w = (T[:3, :3] @ pts.T).T + T[:3, 3]
            over = w[w[:, 0] - radius < table_back_x]
            out[name] = float((over[:, 2] - radius).min() - table_z) if len(over) else float("inf")
        return out
