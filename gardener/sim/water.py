"""Particle water in NVIDIA Warp: position-based fluids (Macklin and Mueller, 2013) on the GPU.

threepp's pip wheel has no PhysX fluid, so this runs on its own, next to PhysX. Each frame the can's
pose goes in and the particles come out. The coupling is one way: the can moves the water, but the
water pushes nothing back, so its weight has to be added to the can's mass.

Water collides with oriented boxes (the can's walls, the table) and open cylinders (the pots). Walls
are thicker than a particle moves in one substep, otherwise particles could tunnel through them.

Lengths in metres, z up.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import warp as wp

wp.config.verbose = False


@dataclass
class Pot:
    x: float
    y: float
    z: float              # the pot's underside
    radius: float = 0.050  # inner radius
    height: float = 0.080
    wall: float = 0.010
    floor: float = 0.010


@dataclass
class WaterParams:
    spacing: float = 0.005      # distance between particles at rest
    substeps: int = 8
    max_gain: float = 0.05      # m/s the density solver may add to a particle in one substep
    max_speed: float = 2.5      # m/s; at 8 substeps and 60 frames/s that is 5 mm a substep, under the 10 mm walls
    iterations: int = 4
    relaxation: float = 200.0   # keeps the density solve stable
    viscosity: float = 0.02     # XSPH: higher is more like honey
    cohesion_k: float = 1e-5    # artificial pressure, which stops particles clumping
    friction: float = 0.5       # how much sliding a surface removes in one contact


@wp.kernel
def _predict(pos: wp.array(dtype=wp.vec3), vel: wp.array(dtype=wp.vec3), pred: wp.array(dtype=wp.vec3), dt: float, max_speed: float):
    i = wp.tid()
    v = vel[i] + wp.vec3(0.0, 0.0, -9.81) * dt
    speed = wp.length(v)
    if speed > max_speed:  # a particle must not cross a wall in one substep
        v = v * (max_speed / speed)
    vel[i] = v
    pred[i] = pos[i] + v * dt


@wp.kernel
def _lambdas(grid: wp.uint64, pred: wp.array(dtype=wp.vec3), lam: wp.array(dtype=float), h: float, rho0: float,
             poly6: float, spiky: float, relax: float):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    x = pred[i]
    rho = float(0.0)
    grad_i = wp.vec3(0.0, 0.0, 0.0)
    sum_sq = float(0.0)
    query = wp.hash_grid_query(grid, x, h)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        d = x - pred[j]
        r2 = wp.dot(d, d)
        if r2 < h * h:
            w = h * h - r2
            rho += poly6 * w * w * w
            r = wp.sqrt(r2)
            if r > 1e-9 and j != i:
                g = d * (spiky * (h - r) * (h - r) / r / rho0)
                grad_i += g
                sum_sq += wp.dot(g, g)
    c = rho / rho0 - 1.0
    if c < 0.0:
        c = 0.0  # no suction: free surfaces stay free
    lam[i] = -c / (sum_sq + wp.dot(grad_i, grad_i) + relax)


@wp.kernel
def _corrections(grid: wp.uint64, pred: wp.array(dtype=wp.vec3), lam: wp.array(dtype=float), delta: wp.array(dtype=wp.vec3),
                 h: float, rho0: float, spiky: float, poly6: float, wq: float, k: float):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    x = pred[i]
    out = wp.vec3(0.0, 0.0, 0.0)
    query = wp.hash_grid_query(grid, x, h)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        if j != i:
            d = x - pred[j]
            r2 = wp.dot(d, d)
            if r2 < h * h:
                r = wp.sqrt(r2)
                if r > 1e-9:
                    w = h * h - r2
                    ratio = poly6 * w * w * w / wq
                    s = -k * ratio * ratio * ratio * ratio
                    out += d * ((lam[i] + lam[j] + s) * spiky * (h - r) * (h - r) / r)
    delta[i] = out / rho0


@wp.kernel
def _apply(pred: wp.array(dtype=wp.vec3), delta: wp.array(dtype=wp.vec3), max_step: float):
    i = wp.tid()
    d = delta[i]
    size = wp.length(d)
    if size > max_step:  # a tight squeeze must not fire a particle through a wall
        d = d * (max_step / size)
    pred[i] = pred[i] + d


@wp.kernel
def _collide_boxes(pred: wp.array(dtype=wp.vec3), pos: wp.array(dtype=wp.vec3), boxes: wp.array(dtype=wp.transform),
                   half: wp.array(dtype=wp.vec3), n: int, radius: float, friction: float):
    i = wp.tid()
    p = pred[i]
    for b in range(n):
        T = boxes[b]
        q = wp.transform_point(wp.transform_inverse(T), p)
        e = half[b] + wp.vec3(radius, radius, radius)
        dx = e[0] - wp.abs(q[0])
        dy = e[1] - wp.abs(q[1])
        dz = e[2] - wp.abs(q[2])
        if dx > 0.0 and dy > 0.0 and dz > 0.0:
            # Out through the nearest face; the last step's motion is damped along that face.
            old = wp.transform_point(wp.transform_inverse(T), pos[i])
            if dx <= dy and dx <= dz:
                q = wp.vec3(wp.sign(q[0]) * e[0], q[1] + (old[1] - q[1]) * friction, q[2] + (old[2] - q[2]) * friction)
            elif dy <= dz:
                q = wp.vec3(q[0] + (old[0] - q[0]) * friction, wp.sign(q[1]) * e[1], q[2] + (old[2] - q[2]) * friction)
            else:
                q = wp.vec3(q[0] + (old[0] - q[0]) * friction, q[1] + (old[1] - q[1]) * friction, wp.sign(q[2]) * e[2])
            p = wp.transform_point(T, q)
    pred[i] = p


@wp.kernel
def _collide_pots(pred: wp.array(dtype=wp.vec3), pots: wp.array(dtype=wp.vec3), shape: wp.array(dtype=wp.vec4), n: int, radius: float):
    """Each pot: a floor disc and a wall ring, open at the top."""
    i = wp.tid()
    p = pred[i]
    for k in range(n):
        c = pots[k]
        s = shape[k]  # inner radius, height, wall, floor
        dx = p[0] - c[0]
        dy = p[1] - c[1]
        r = wp.sqrt(dx * dx + dy * dy)
        top = c[2] + s[1]
        outer = s[0] + s[2]
        if p[2] < top + radius and p[2] > c[2] - radius and r < outer + radius:
            floor_top = c[2] + s[3]
            if r > 1e-6 and r > s[0] - radius:
                # In the wall ring: out through the inner or the outer face, or up over the rim.
                to_in = r - (s[0] - radius)
                to_out = (outer + radius) - r
                to_top = top + radius - p[2]
                if to_top < to_in and to_top < to_out:
                    p = wp.vec3(p[0], p[1], top + radius)
                elif to_in < to_out:
                    f = (s[0] - radius) / r
                    p = wp.vec3(c[0] + dx * f, c[1] + dy * f, p[2])
                    r = s[0] - radius
                else:
                    f = (outer + radius) / r
                    p = wp.vec3(c[0] + dx * f, c[1] + dy * f, p[2])
                    r = outer + radius
            # Checked after the wall, so a particle right beside the wall still meets the floor.
            if r <= s[0] and p[2] < floor_top + radius:
                p = wp.vec3(p[0], p[1], floor_top + radius)
    pred[i] = p


@wp.kernel
def _finish(pos: wp.array(dtype=wp.vec3), pred: wp.array(dtype=wp.vec3), vel: wp.array(dtype=wp.vec3), dt: float, ground: float,
            radius: float, max_gain: float):
    i = wp.tid()
    p = pred[i]
    if p[2] < ground + radius:
        p = wp.vec3(p[0], p[1], ground + radius)
    # Velocity comes from how far the solver moved the particle. A big correction in a short step would
    # read as a huge speed and make water explode off a floor, so the solver may add only a little.
    v = (p - pos[i]) / dt
    limit = wp.length(vel[i]) + max_gain
    speed = wp.length(v)
    if speed > limit:
        v = v * (limit / speed)
    vel[i] = v
    pos[i] = p


@wp.kernel
def _viscosity(grid: wp.uint64, pos: wp.array(dtype=wp.vec3), vel: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.vec3),
               h: float, rho0: float, poly6: float, c: float):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    x = pos[i]
    v = vel[i]
    dv = wp.vec3(0.0, 0.0, 0.0)
    query = wp.hash_grid_query(grid, x, h)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        if j != i:
            d = x - pos[j]
            r2 = wp.dot(d, d)
            if r2 < h * h:
                w = h * h - r2
                dv += (vel[j] - v) * (poly6 * w * w * w)
    out[i] = v + dv * (c / rho0)


def fill_lattice(half_x: float, half_y: float, height: float, spacing: float, count: int) -> np.ndarray:
    """Up to `count` points on a lattice in a box, filled from the bottom. Local frame, base at z = 0."""
    nx, ny = int(2 * half_x / spacing), int(2 * half_y / spacing)
    nz = int(height / spacing)
    pts = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                if len(pts) == count:
                    return np.array(pts)
                pts.append(((i + 0.5) * spacing - half_x, (j + 0.5) * spacing - half_y, (k + 0.5) * spacing))
    return np.array(pts)


class WaterSim:
    def __init__(self, points: np.ndarray, pots: list[Pot], static_boxes: list, params: WaterParams | None = None,
                 ground: float = 0.0, device: str | None = None):
        """static_boxes: list of (centre xyz, half sizes xyz), axis-aligned, such as the table top."""
        self.p = params or WaterParams()
        self.device = device or ("cuda:0" if wp.is_cuda_available() else "cpu")
        self.n = len(points)
        self.pots = pots
        self.static_boxes = static_boxes
        self.ground = ground
        self.can_boxes: list = []
        s = self.p.spacing
        self.radius = 0.5 * s
        self.h = 2.0 * s
        h = self.h
        self.poly6 = 315.0 / (64.0 * math.pi * h ** 9)
        self.spiky = -45.0 / (math.pi * h ** 6)
        self.wq = self.poly6 * (h * h - (0.2 * h) ** 2) ** 3
        # Rest density: the density a particle has on the rest lattice.
        rho0 = 0.0
        m = int(math.ceil(h / s))
        for i in range(-m, m + 1):
            for j in range(-m, m + 1):
                for k in range(-m, m + 1):
                    r2 = (i * i + j * j + k * k) * s * s
                    if r2 < h * h:
                        rho0 += self.poly6 * (h * h - r2) ** 3
        self.rho0 = rho0
        d = self.device
        self.pos = wp.array(points.astype(np.float32), dtype=wp.vec3, device=d)
        self.vel = wp.zeros(self.n, dtype=wp.vec3, device=d)
        self.pred = wp.zeros(self.n, dtype=wp.vec3, device=d)
        self.tmp = wp.zeros(self.n, dtype=wp.vec3, device=d)
        self.delta = wp.zeros(self.n, dtype=wp.vec3, device=d)
        self.lam = wp.zeros(self.n, dtype=float, device=d)
        self.grid = wp.HashGrid(128, 128, 128, device=d)
        self.pot_c = wp.array(np.array([[q.x, q.y, q.z] for q in pots], dtype=np.float32).reshape(-1, 3), dtype=wp.vec3, device=d)
        self.pot_s = wp.array(np.array([[q.radius, q.height, q.wall, q.floor] for q in pots], dtype=np.float32).reshape(-1, 4), dtype=wp.vec4, device=d)
        self._box_T = None
        self._box_h = None

    def set_can(self, T_can: np.ndarray, boxes_local: list) -> None:
        """T_can: 4x4 pose. boxes_local: list of (centre xyz, half sizes xyz) in the can's frame."""
        q = _matrix_to_quat(T_can[:3, :3])
        tf, half = [], []
        for centre, h in boxes_local:
            c = T_can[:3, :3] @ np.asarray(centre) + T_can[:3, 3]
            tf.append([*c, *q])
            half.append(h)
        for centre, h in self.static_boxes:
            tf.append([*centre, 0.0, 0.0, 0.0, 1.0])
            half.append(h)
        self._box_T = wp.array(np.array(tf, dtype=np.float32), dtype=wp.transform, device=self.device)
        self._box_h = wp.array(np.array(half, dtype=np.float32), dtype=wp.vec3, device=self.device)

    def step(self, frame_dt: float) -> None:
        p, d, n = self.p, self.device, self.n
        dt = frame_dt / p.substeps
        nb = self._box_T.shape[0]
        for _ in range(p.substeps):
            wp.launch(_predict, n, (self.pos, self.vel, self.pred, dt, p.max_speed), device=d)
            self.grid.build(self.pred, self.h)
            for _ in range(p.iterations):
                wp.launch(_lambdas, n, (self.grid.id, self.pred, self.lam, self.h, self.rho0, self.poly6, self.spiky, p.relaxation), device=d)
                wp.launch(_corrections, n, (self.grid.id, self.pred, self.lam, self.delta, self.h, self.rho0, self.spiky, self.poly6, self.wq, p.cohesion_k), device=d)
                wp.launch(_apply, n, (self.pred, self.delta, 0.2 * p.spacing), device=d)
                wp.launch(_collide_boxes, n, (self.pred, self.pos, self._box_T, self._box_h, nb, self.radius, p.friction), device=d)
                wp.launch(_collide_pots, n, (self.pred, self.pot_c, self.pot_s, len(self.pots), self.radius), device=d)
            wp.launch(_finish, n, (self.pos, self.pred, self.vel, dt, self.ground, self.radius, p.max_gain), device=d)
            self.grid.build(self.pos, self.h)
            wp.launch(_viscosity, n, (self.grid.id, self.pos, self.vel, self.tmp, self.h, self.rho0, self.poly6, p.viscosity), device=d)
            self.vel, self.tmp = self.tmp, self.vel

    def positions(self) -> np.ndarray:
        return self.pos.numpy()

    def count(self, T_can: np.ndarray, can_half_x: float, can_half_y: float, can_height: float, table_z: float | None = None) -> dict:
        """How many particles are in each place. Everything is counted exactly once."""
        x = self.positions()
        left = np.ones(self.n, dtype=bool)
        out = {}
        local = (x - T_can[:3, 3]) @ T_can[:3, :3]
        in_can = (np.abs(local[:, 0]) < can_half_x) & (np.abs(local[:, 1]) < can_half_y) & (local[:, 2] > 0) & (local[:, 2] < can_height)
        out["can"] = int(in_can.sum())
        left &= ~in_can
        for k, q in enumerate(self.pots):
            r = np.hypot(x[:, 0] - q.x, x[:, 1] - q.y)
            inside = left & (r < q.radius) & (x[:, 2] > q.z) & (x[:, 2] < q.z + q.height)
            out[f"pot{k}"] = int(inside.sum())
            left &= ~inside
        if table_z is not None:
            on = left & (x[:, 2] >= table_z - 1e-3) & (x[:, 2] < table_z + 0.02)
            out["table"] = int(on.sum())
            left &= ~on
            below = left & (x[:, 2] < table_z - 1e-3)
            out["floor"] = int(below.sum())
            left &= ~below
        out["air"] = int(left.sum())
        return out


def _matrix_to_quat(R: np.ndarray) -> list:
    """Rotation matrix to (x, y, z, w)."""
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        return [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s]
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
    q = [0.0, 0.0, 0.0, (R[k, j] - R[j, k]) / s]
    q[i] = 0.25 * s
    q[j] = (R[j, i] + R[i, j]) / s
    q[k] = (R[k, i] + R[i, k]) / s
    return q


def can_water_boxes(half_x: float, half_y: float, height: float, wall: float = 0.010, spout_drop: float = 0.025) -> list:
    """The can's walls as boxes, in the can's frame: origin at the floor's centre, z up, +x is the pour side.

    half_x, half_y and height describe the inside. The +x wall is lower: that notch is the spout.
    """
    t = wall / 2
    hx, hy = half_x + wall, half_y + wall
    return [
        ((0.0, 0.0, -t), (hx, hy, t)),                                                        # floor
        ((-half_x - t, 0.0, height / 2), (t, hy, height / 2)),                                # back
        ((half_x + t, 0.0, (height - spout_drop) / 2), (t, hy, (height - spout_drop) / 2)),   # front, with the spout
        ((0.0, -half_y - t, height / 2), (hx, t, height / 2)),                                # sides
        ((0.0, half_y + t, height / 2), (hx, t, height / 2)),
    ]
