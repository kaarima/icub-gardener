#!/usr/bin/env python3
"""Make the "gardener" version of the iCub URDF.

What it does (the original file is never modified):
  1. LOCKS every joint we don't need by shrinking its range to almost zero
     (+/- 1e-4 rad), so it stays at its zero angle but keeps its own mesh.
     Kept movable: torso (3), right arm (7), right hand (20) = 30 joints.
     Locked: legs, neck, eyes, left arm, left hand (45 joints).
  2. LIFTS the robot: adds a `world` link under iCub's waist, so his waist
     sits `--height` metres above the floor instead of at floor level.

Usage:
  python scripts/make_gardener_urdf.py \
      --src ~/icub-assets/iCub/robots/iCubGazeboV2_10/model_abs.urdf \
      --out ~/icub-assets/iCub/robots/iCubGazeboV2_10/gardener.urdf \
      --height 0.6
"""
import argparse
import os
import re
import xml.etree.ElementTree as ET

MOVABLE_TYPES = {"revolute", "continuous", "prismatic"}
LOCK_EPS = 1e-4  # rad: locked joints may move at most +/- this much

# Joints that stay movable. Everything else movable gets locked.
KEEP_EXACT = {
    "torso_pitch", "torso_roll", "torso_yaw",
    "r_shoulder_pitch", "r_shoulder_roll", "r_shoulder_yaw",
    "r_elbow",
    "r_wrist_prosup", "r_wrist_pitch", "r_wrist_yaw",
}
KEEP_PATTERN = re.compile(r"^r_hand_(thumb|index|middle|ring|little)_[0-3]_joint$")


def keep(name: str) -> bool:
    return name in KEEP_EXACT or bool(KEEP_PATTERN.match(name))


def find_root_link(robot: ET.Element) -> str:
    links = {l.get("name") for l in robot.findall("link")}
    children = {j.find("child").get("link") for j in robot.findall("joint")}
    roots = sorted(links - children)
    if len(roots) != 1:
        raise RuntimeError(f"expected exactly one root link, found {roots}")
    return roots[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="input URDF (absolute mesh paths)")
    ap.add_argument("--out", required=True, help="output URDF")
    ap.add_argument("--height", type=float, default=0.6, help="waist height above the floor, metres")
    ap.add_argument("--lock-mode", choices=["limits", "fixed"], default="limits",
                    help="limits (default): near-zero joint range, every part stays visible; "
                         "fixed: fixed joints (threepp then hides the merged parts)")
    args = ap.parse_args()

    src = os.path.expanduser(args.src)
    out = os.path.expanduser(args.out)

    tree = ET.parse(src)
    robot = tree.getroot()

    kept, locked = [], []
    for j in robot.findall("joint"):
        name, jtype = j.get("name"), j.get("type")
        if jtype not in MOVABLE_TYPES:
            continue
        if keep(name):
            kept.append(name)
            continue
        if args.lock_mode == "fixed":
            # Fixed joint at its zero angle. NOTE: threepp merges links joined
            # by fixed joints and keeps only one visual per merged link, so the
            # locked parts become invisible. Kept only as an option.
            j.set("type", "fixed")
            for tag in ("axis", "limit", "dynamics", "safety_controller", "mimic", "calibration"):
                for el in j.findall(tag):
                    j.remove(el)
        else:
            # Default: keep the joint (so its link keeps its own mesh) but give
            # it an almost-zero range, so it cannot move.
            if jtype == "continuous":
                j.set("type", "revolute")
            lim = j.find("limit")
            if lim is None:
                lim = ET.SubElement(j, "limit", {"effort": "100", "velocity": "1"})
            lim.set("lower", f"{-LOCK_EPS}")
            lim.set("upper", f"{LOCK_EPS}")
            for el in j.findall("mimic"):
                j.remove(el)
        locked.append(name)

    # Lift: new root link "world" + fixed joint to iCub's old root (the waist).
    # URDF is z-up, so the lift goes along z.
    old_root = find_root_link(robot)
    world = ET.Element("link", {"name": "world"})
    inertial = ET.SubElement(world, "inertial")
    ET.SubElement(inertial, "mass", {"value": "0.01"})
    ET.SubElement(inertial, "inertia", {"ixx": "1e-5", "iyy": "1e-5", "izz": "1e-5",
                                        "ixy": "0", "ixz": "0", "iyz": "0"})
    robot.insert(0, world)
    lift = ET.Element("joint", {"name": "world_to_" + old_root, "type": "fixed"})
    ET.SubElement(lift, "parent", {"link": "world"})
    ET.SubElement(lift, "child", {"link": old_root})
    ET.SubElement(lift, "origin", {"xyz": f"0 0 {args.height}", "rpy": "0 0 0"})
    robot.insert(1, lift)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    tree.write(out, encoding="utf-8", xml_declaration=True)

    hand = [n for n in kept if n.startswith("r_hand_")]
    print(f"root link: {old_root}  (lifted {args.height} m)")
    print(f"kept movable: {len(kept)}  (torso+arm {len(kept) - len(hand)}, right hand {len(hand)})")
    print(f"locked: {len(locked)}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()