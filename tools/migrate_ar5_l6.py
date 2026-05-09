#!/usr/bin/env python3
"""One-off migration: split legacy `AR5_L6_description/` into components.

The legacy layout has arm + hand combined in one URDF per side, plus a
separate workstation URDF. This script reorganizes those into the new
component structure under `assets/components/`:

    assets/components/arms/ar5/variants/{left,right}/{arm.urdf, meshes/}
    assets/components/hands/linkerhand_l6/variants/{left,right}/{hand.urdf, meshes/}
    assets/components/bases/bench_table/variants/default/{base.urdf, meshes/}

The legacy files under `assets/urdf/` are NOT deleted by this script; they
remain in place so the existing Isaac-side runtime keeps working during
the PR #1 → PR #2 transition.

Usage:
    python -m tools.migrate_ar5_l6
"""

from __future__ import annotations

import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LEGACY_URDF_DIR = REPO_ROOT / "assets" / "urdf"
LEGACY_AR5_L6_DIR = LEGACY_URDF_DIR / "AR5_L6_description"
COMPONENTS_DIR = REPO_ROOT / "assets" / "components"


def _write_tree(root: ET.Element, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Indent for readability; the composer re-normalizes output anyway.
    ET.indent(root, space="  ", level=0)
    tree = ET.ElementTree(root)
    tree.write(str(path), encoding="UTF-8", xml_declaration=True)
    # ET writes `'utf-8'`; uppercase it to match our composer output.
    text = path.read_text()
    text = text.replace(
        "<?xml version='1.0' encoding='utf-8'?>",
        '<?xml version="1.0" encoding="UTF-8"?>',
    )
    if not text.endswith("\n"):
        text += "\n"
    path.write_text(text)


def _rewrite_mesh_paths(root: ET.Element, prefix_old: str, prefix_new: str) -> None:
    for mesh in root.iter("mesh"):
        f = mesh.attrib.get("filename", "")
        if f.startswith(prefix_old):
            mesh.attrib["filename"] = prefix_new + f[len(prefix_old) :]


def _rewrite_mujoco_meshdir(root: ET.Element, new_meshdir: str) -> None:
    for comp in root.iter("compiler"):
        if comp.attrib.get("meshdir"):
            comp.attrib["meshdir"] = new_meshdir


def split_combined_urdf(
    src: Path,
    *,
    arm_link_prefix: str,
    hand_link_prefix: str,
    arm_out: Path,
    hand_out: Path,
    arm_mesh_src_prefix: str,
    hand_mesh_src_prefix: str,
) -> None:
    """Split a combined arm+hand URDF into two component URDFs.

    Partition rule: elements are assigned to arm or hand by the name prefix
    of the link(s) they reference. Materials are duplicated into both
    (composer will role-prefix them). The `world` link, the `fixed` joint
    attaching world to arm_base, and the `arm_hand_fixed` joint are
    dropped — the composer re-introduces them via `freeze_base` and
    recipe mount entries respectively. `<gazebo>` blocks are dropped.
    `<mujoco>` goes with the hand (all equality constraints are on hand
    joints).
    """
    tree = ET.parse(str(src))
    src_root = tree.getroot()

    arm_root = ET.Element("robot", {"name": arm_out.stem})
    hand_root = ET.Element("robot", {"name": hand_out.stem})

    # 1. Copy materials into both.
    for mat in src_root.findall("material"):
        arm_root.append(_deep_copy(mat))
        hand_root.append(_deep_copy(mat))

    def _is_arm_name(name: str) -> bool:
        return name.startswith(arm_link_prefix)

    def _is_hand_name(name: str) -> bool:
        return name.startswith(hand_link_prefix)

    # 2. Links.
    for link in src_root.findall("link"):
        name = link.attrib.get("name", "")
        if name == "world":
            continue
        if _is_arm_name(name):
            arm_root.append(_deep_copy(link))
        elif _is_hand_name(name):
            hand_root.append(_deep_copy(link))
        else:
            print(f"  warn: unassigned link {name!r} — dropped", file=sys.stderr)

    # 3. Joints. Skip the world→base fixed joint and the arm→hand splice.
    for joint in src_root.findall("joint"):
        jname = joint.attrib.get("name", "")
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        pl = parent.attrib.get("link", "")
        cl = child.attrib.get("link", "")
        if pl == "world":
            continue  # world→base fixed
        if jname == "arm_hand_fixed":
            continue  # splice joint — recipe handles mount
        # Joint goes wherever its child lives.
        if _is_arm_name(cl):
            arm_root.append(_deep_copy(joint))
        elif _is_hand_name(cl):
            hand_root.append(_deep_copy(joint))
        else:
            print(f"  warn: unassigned joint {jname!r} — dropped", file=sys.stderr)

    # 4. Transmissions. Partition by the joint they drive.
    for trans in src_root.findall("transmission"):
        driven = None
        sub = trans.find("joint")
        if sub is not None:
            driven = sub.attrib.get("name", "")
        if driven and _is_arm_name(driven):
            arm_root.append(_deep_copy(trans))
        elif driven and _is_hand_name(driven):
            hand_root.append(_deep_copy(trans))
        else:
            # Legacy AR5 URDFs have transmissions referencing broken joint
            # names like `xmate_joint_*` or `AR5_5_07L_W4C4A2_joint_8`.
            # Assign them to arm since they're part of that vendor's
            # transmission table even if the references are stale.
            arm_root.append(_deep_copy(trans))

    # 5. <mujoco> block → hand only.
    for mj in src_root.findall("mujoco"):
        hand_root.append(_deep_copy(mj))

    # 6. <gazebo> blocks → dropped (composer drops them too).

    # 7. Rewrite mesh paths to `meshes/<file>`.
    _rewrite_mesh_paths(arm_root, arm_mesh_src_prefix, "meshes/")
    _rewrite_mesh_paths(hand_root, hand_mesh_src_prefix, "meshes/")
    _rewrite_mujoco_meshdir(hand_root, "meshes/")

    _write_tree(arm_root, arm_out)
    _write_tree(hand_root, hand_out)


def _deep_copy(elem: ET.Element) -> ET.Element:
    out = ET.Element(elem.tag, dict(elem.attrib))
    out.text = elem.text
    out.tail = elem.tail
    for child in elem:
        out.append(_deep_copy(child))
    return out


def copy_meshes(src_dir: Path, dst_dir: Path) -> int:
    """Copy every file under src_dir into dst_dir. Returns count."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in src_dir.iterdir():
        if p.is_file():
            shutil.copy2(p, dst_dir / p.name)
            n += 1
    return n


def migrate_bench_table() -> None:
    """Lift `workstation.urdf` + its STL meshes into `bases/bench_table/`.

    Also adds a `workstation_arm_left_mount` and `workstation_arm_right_mount`
    link to provide the recipe-referenced mount frames. These are empty
    placeholder links welded to the table at the same offsets previously
    hardcoded in `sim/assets/scene_assets.py`; the composer's mount joint
    then attaches an arm's root link to one of these via a fixed joint.
    """
    src_urdf = LEGACY_URDF_DIR / "workstation.urdf"
    out_dir = COMPONENTS_DIR / "bases" / "bench_table" / "variants" / "default"
    out_urdf = out_dir / "base.urdf"
    out_meshes = out_dir / "meshes"

    tree = ET.parse(str(src_urdf))
    root = tree.getroot()

    # Copy meshes (workstation.STL, camera.STL, camera_base_link.STL).
    out_meshes.mkdir(parents=True, exist_ok=True)
    for name in ("workstation.STL", "camera.STL", "camera_base_link.STL"):
        src = LEGACY_URDF_DIR / name
        if src.is_file():
            shutil.copy2(src, out_meshes / name)

    # Rewrite mesh paths: they were bare `workstation.STL` etc.
    for mesh in root.iter("mesh"):
        fn = mesh.attrib.get("filename", "")
        if fn and not fn.startswith(("package://", "/", "http://", "https://", "meshes/")):
            mesh.attrib["filename"] = f"meshes/{fn}"

    # Add empty mount-point links welded to the table.
    # Offsets previously hardcoded in sim/assets/scene_assets.py as:
    #   left:  (0.0637, 0.719, 1.267), quat (0.5, -0.5, 0.5, 0.5)
    #   right: (0.0637, 0.536, 1.267), quat (0.5,  0.5, 0.5, -0.5)
    # Quat -> rpy (URDF XYZ extrinsic): see comment below.
    import math
    PI2 = math.pi / 2
    mounts = [
        ("workstation_arm_left_mount",  (0.0637, 0.719, 1.267), (0.0, PI2,  PI2)),
        ("workstation_arm_right_mount", (0.0637, 0.536, 1.267), (0.0, PI2, -PI2)),
    ]
    for mount_name, xyz, rpy in mounts:
        link = ET.Element("link", {"name": mount_name})
        # Empty link (no inertial / visual / collision) — MuJoCo/Isaac
        # treat unsupported empty URDF links fine when attached via fixed
        # joint to a parent. If this causes trouble in any loader, swap
        # for a 1e-4 box like the legacy `world`/`tcp` placeholders.
        root.append(link)
        j = ET.Element(
            "joint",
            {"name": f"weld_{mount_name}", "type": "fixed"},
        )
        ET.SubElement(j, "origin", {
            "xyz": " ".join(f"{v:.9g}" for v in xyz),
            "rpy": " ".join(f"{v:.9g}" for v in rpy),
        })
        ET.SubElement(j, "parent", {"link": "workstation_link"})
        ET.SubElement(j, "child", {"link": mount_name})
        root.append(j)

    _write_tree(root, out_urdf)
    print(f"  wrote {out_urdf.relative_to(REPO_ROOT)}")


def migrate_arm_and_hand() -> None:
    """Split the combined AR5_L6 URDFs into arm and hand components."""
    arm_root_dir = COMPONENTS_DIR / "arms" / "ar5" / "variants"
    hand_root_dir = COMPONENTS_DIR / "hands" / "linkerhand_l6" / "variants"

    sides = [
        ("left", "AR5_L6_left.urdf", "AR5_5_07L_W4C4A2_", "lh_", "AR5_left_meshes/", "L6_left_meshes/", "AR5_left_meshes", "L6_left_meshes"),
        ("right", "AR5_L6_right.urdf", "AR5_5_07R_W4C4A2_", "rh_", "AR5_right_meshes/", "L6_right_meshes/", "AR5_right_meshes", "L6_right_meshes"),
    ]
    for side, fname, arm_pref, hand_pref, arm_mesh_pref, hand_mesh_pref, arm_mesh_dir, hand_mesh_dir in sides:
        src = LEGACY_AR5_L6_DIR / fname
        arm_out_dir = arm_root_dir / side
        hand_out_dir = hand_root_dir / side

        split_combined_urdf(
            src=src,
            arm_link_prefix=arm_pref,
            hand_link_prefix=hand_pref,
            arm_out=arm_out_dir / "arm.urdf",
            hand_out=hand_out_dir / "hand.urdf",
            arm_mesh_src_prefix=arm_mesh_pref,
            hand_mesh_src_prefix=hand_mesh_pref,
        )
        n_arm = copy_meshes(LEGACY_AR5_L6_DIR / arm_mesh_dir, arm_out_dir / "meshes")
        n_hand = copy_meshes(LEGACY_AR5_L6_DIR / hand_mesh_dir, hand_out_dir / "meshes")
        print(f"  {side}: arm.urdf + {n_arm} arm meshes, hand.urdf + {n_hand} hand meshes")


def main() -> int:
    print("[migrate] arm + hand ...")
    migrate_arm_and_hand()
    print("[migrate] bench_table ...")
    migrate_bench_table()
    print("[migrate] done. next: write meta.yaml files and a recipe.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
