#!/usr/bin/env python3
"""Pour test with no robot (the book's Phase 2): the script moves the can over a pot and tilts it.

Checks that the water stays in the can when still, pours into the pot, is counted correctly, and does
not leak through walls. Prints VRAM and simulation speed.

Usage:
  python scripts/test_pour.py --particles 2000
  python scripts/test_pour.py --particles 2000 --no-tilt    # control: nothing may leave the can

Nothing is written to disk.
"""
import argparse
import math
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gardener.sim.water import Pot, WaterParams, WaterSim, can_water_boxes, fill_lattice  # noqa: E402

FRAME = 1.0 / 60.0
TABLE_Z = 0.58
INNER_HALF = 0.025     # the can's inside: 50 x 50 mm, as walls take 10 mm from the 70 mm body
INNER_HEIGHT = 0.085


def vram_mb():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return int(out.stdout.strip().splitlines()[0])


def can_pose(position, tilt):
    """Upright at tilt 0. A positive tilt turns the +x (spout) side down, about the y axis."""
    c, s = math.cos(tilt), math.sin(tilt)
    T = np.eye(4)
    T[:3, :3] = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    T[:3, 3] = position
    return T


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--particles", type=int, default=2000)
    ap.add_argument("--spacing", type=float, default=None, help="m; default: fills 60%% of the can with this many particles")
    ap.add_argument("--tilt", type=float, default=75.0, help="deg")
    ap.add_argument("--no-tilt", action="store_true")
    ap.add_argument("--hold", type=float, default=4.0, help="seconds at full tilt")
    ap.add_argument("--ramp", type=float, default=2.0, help="seconds to reach full tilt")
    ap.add_argument("--aim", type=float, default=-0.075, help="can position relative to the pot centre along x (m)")
    ap.add_argument("--above", type=float, default=0.03, help="can floor above the pot's rim (m)")
    args = ap.parse_args()

    # The book asks for 40-90% full; the can's volume then sets the particle spacing.
    spacing = args.spacing or ((2 * INNER_HALF) ** 2 * INNER_HEIGHT * 0.6 / args.particles) ** (1 / 3)
    base = vram_mb()
    pot = Pot(x=-0.40, y=0.10, z=TABLE_Z)
    other = Pot(x=-0.40, y=-0.15, z=TABLE_Z)
    table = ((-0.41, 0.0, TABLE_Z - 0.015), (0.30, 0.40, 0.015))
    boxes = can_water_boxes(INNER_HALF, INNER_HALF, INNER_HEIGHT)

    # Above the pot, with the spout side over the pot's middle when tilted.
    rim = pot.z + pot.height
    start = np.array([pot.x + args.aim, pot.y, rim + args.above])
    local = fill_lattice(INNER_HALF - spacing / 2, INNER_HALF - spacing / 2, INNER_HEIGHT - 0.002, spacing, args.particles)
    points = local + start
    sim = WaterSim(points, [pot, other], [table], WaterParams(spacing=spacing), ground=0.0)
    print(f"{sim.n} particles, spacing {spacing * 1000:.2f} mm, radius {sim.radius * 1000:.2f} mm, on {sim.device}")
    print(f"can {(sim.n * spacing ** 3) / ((2 * INNER_HALF) ** 2 * INNER_HEIGHT) * 100:.0f}% full")

    tilt_rad = 0.0 if args.no_tilt else math.radians(args.tilt)
    t, stepped, phys = 0.0, 0, 0.0
    settle, ramp, back, after = 1.5, args.ramp, 1.5, 2.0
    T_total = settle + ramp + args.hold + back + after
    peak_vram = base
    log_at = (settle, settle + ramp, settle + ramp + args.hold, settle + ramp + args.hold + back, T_total)
    next_log = 0

    def counts(T):
        return sim.count(T, INNER_HALF + 0.012, INNER_HALF + 0.012, INNER_HEIGHT + 0.005, TABLE_Z)

    ever_escaped = 0
    while t < T_total:
        if t < settle:
            a = 0.0
        elif t < settle + ramp:
            u = (t - settle) / ramp
            a = tilt_rad * u * u * (3 - 2 * u)
        elif t < settle + ramp + args.hold:
            a = tilt_rad
        elif t < settle + ramp + args.hold + back:
            u = (t - settle - ramp - args.hold) / back
            a = tilt_rad * (1 - u * u * (3 - 2 * u))
        else:
            a = 0.0
        T = can_pose(start, a)
        sim.set_can(T, boxes)
        t0 = time.perf_counter()
        sim.step(FRAME)
        phys += time.perf_counter() - t0
        t += FRAME
        stepped += 1
        if stepped % 30 == 0:
            peak_vram = max(peak_vram, vram_mb())
        if next_log < len(log_at) and t >= log_at[next_log]:
            c = counts(T)
            x = sim.positions()
            print(f"t={t:5.2f}s tilt {math.degrees(a):5.1f} deg | " + ", ".join(f"{k} {v}" for k, v in c.items())
                  + f" | total {sum(c.values())}, max speed {np.abs(sim.vel.numpy()).max():.2f} m/s, nan {int(np.isnan(x).sum())}")
            next_log += 1

    T = can_pose(start, 0.0)
    c = counts(T)
    left_can = sim.n - c["can"]
    in_target = c["pot0"]
    print(f"\nRESULT: {left_can} of {sim.n} left the can; in the target pot {in_target} "
          f"({100 * in_target / max(left_can, 1):.0f}%), other pot {c['pot1']}, table {c['table']}, floor {c['floor']}, air {c['air']}")
    # Leak check: anything inside the pot's footprint but below its floor came through the wall.
    x = sim.positions()
    r = np.hypot(x[:, 0] - pot.x, x[:, 1] - pot.y)
    in_wall = int(((r > pot.radius) & (r < pot.radius + pot.wall) & (x[:, 2] > pot.z) & (x[:, 2] < rim)).sum())
    under = int(((r < pot.radius) & (x[:, 2] < pot.z + pot.floor)).sum())
    print(f"leak check: {in_wall} particles inside the pot's wall, {under} through its floor; count sums to {sum(c.values())} of {sim.n}")
    spilled = x[(np.hypot(x[:, 0] - pot.x, x[:, 1] - pot.y) >= pot.radius) & (x[:, 2] < TABLE_Z + 0.02)]
    if len(spilled):
        print(f"spill: {len(spilled)} particles, centre {np.round((spilled.mean(0)[:2] - [pot.x, pot.y]) * 1000)} mm from the pot centre, "
              f"x range {np.round((np.percentile(spilled[:, 0], [5, 95]) - pot.x) * 1000)} mm")
    print(f"SPEED: {T_total / phys:.2f} x real time ({stepped} frames, {phys:.1f} s of physics)")
    print(f"VRAM: {peak_vram - base} MiB above the {base} MiB in use before starting")


if __name__ == "__main__":
    main()
