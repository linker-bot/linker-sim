#!/usr/bin/env python3
"""Validate a workstation's composed artifacts.

Checks performed:
  1. Manifest self-consistency: recipe/component hashes match on-disk sources,
     committed workstation.urdf matches manifest.artifacts.urdf_sha256.
  2. Kinematic structure: URDF has expected actuated-joint counts per role;
     declared EE link and every mount frame resolve to existing links.
  3. Mesh resolution: every `<mesh filename=>` path resolves on disk.
  4. Drift: re-run composer in memory; committed artifacts match fresh output.
  5. MJCF parity (when workstation.mjcf is committed):
     - Loads in MuJoCo with no warnings.
     - Actuator-joint order matches manifest.joints[role] (concatenated).
     - Every manifest.frames[role:frame] site at qpos=0 matches the
       corresponding URDF link's world pose to 1e-5 m / 1e-5 rad.

Exit codes:
  0 — all checks pass
  1 — one or more checks failed
  2 — usage / structural error
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import yaml

# Relative imports when invoked as a script-module.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.composer.compose import (  # noqa: E402
    check_drift,
    compose,
    resolve_paths,
    sha256_component_sources,
    sha256_file,
)
from tools.composer.schemas import (  # noqa: E402
    ComponentMeta,
    Recipe,
    SchemaError,
    resolve_variant,
)
from tools.validate_component_mjcf import urdf_link_world_poses  # noqa: E402

try:
    import mujoco
except ImportError:
    mujoco = None  # MJCF checks become SKIP if mujoco isn't installed.


# --------------------------- Check framework ------------------------------ #


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""


def _run(checks: list[tuple[str, Callable[[], str]]]) -> list[CheckResult]:
    """Run named checks. A check returns '' on pass or a message on fail."""
    results: list[CheckResult] = []
    for name, fn in checks:
        try:
            msg = fn()
        except Exception as e:
            results.append(CheckResult(name=name, ok=False, detail=f"exception: {e}"))
            continue
        results.append(CheckResult(name=name, ok=not msg, detail=msg))
    return results


# --------------------------- Individual checks ---------------------------- #


def _check_manifest_exists(paths) -> str:
    if not paths.out_manifest.is_file():
        return f"missing {paths.out_manifest}"
    return ""


def _check_urdf_exists(paths) -> str:
    if not paths.out_urdf.is_file():
        return f"missing {paths.out_urdf}"
    return ""


def _check_recipe_hash(paths, manifest: dict) -> str:
    expected = manifest.get("recipe_sha256")
    if not expected:
        return "manifest has no recipe_sha256"
    actual = sha256_file(paths.recipe)
    if actual != expected:
        return (
            f"recipe.yaml sha256 mismatch: manifest={expected[:12]}…, "
            f"disk={actual[:12]}… (recipe edited without re-running compose)"
        )
    return ""


def _check_component_hashes(paths, recipe: Recipe, manifest: dict) -> str:
    comp_manifest = manifest.get("components", {})
    mismatches: list[str] = []
    for role, cref in recipe.components.items():
        entry = comp_manifest.get(role)
        if entry is None:
            mismatches.append(f"role {role} not in manifest")
            continue
        meta = ComponentMeta.load(paths.components_root / cref.component / "meta.yaml")
        variant = resolve_variant(meta, cref.variant)
        actual = sha256_component_sources(meta, variant)
        if actual != entry.get("sha256"):
            mismatches.append(
                f"{role} ({cref.component}/{variant.name}): sha256 mismatch "
                f"(source edited without re-running compose)"
            )
    return "; ".join(mismatches)


def _check_urdf_hash(paths, manifest: dict) -> str:
    expected = manifest.get("artifacts", {}).get("urdf_sha256")
    if not expected:
        return "manifest has no artifacts.urdf_sha256"
    actual = hashlib.sha256(paths.out_urdf.read_bytes()).hexdigest()
    if actual != expected:
        return "workstation.urdf sha256 mismatch with manifest"
    return ""


def _parse_urdf(paths):
    return ET.parse(str(paths.out_urdf)).getroot()


def _check_joint_counts(paths, manifest: dict) -> str:
    root = _parse_urdf(paths)
    urdf_joints = {j.attrib.get("name", "") for j in root.findall("joint") if j.attrib.get("type") != "fixed"}
    mismatches: list[str] = []
    for role, names in manifest.get("joints", {}).items():
        missing = [n for n in names if n not in urdf_joints]
        if missing:
            mismatches.append(f"{role}: missing in URDF {missing[:3]}{'…' if len(missing) > 3 else ''}")
    return "; ".join(mismatches)


def _check_ee_and_mounts(paths, manifest: dict) -> str:
    root = _parse_urdf(paths)
    link_names = {l.attrib.get("name", "") for l in root.findall("link")}
    missing: list[str] = []
    ee = manifest.get("ee_link")
    if ee and ee not in link_names:
        missing.append(f"ee_link '{ee}' not a link")
    base = manifest.get("base_link")
    if base and base not in link_names:
        missing.append(f"base_link '{base}' not a link")
    for fname, linkname in manifest.get("frames", {}).items():
        if linkname not in link_names:
            missing.append(f"frame '{fname}' -> '{linkname}' not a link")
    return "; ".join(missing)


def _check_mesh_resolution(paths) -> str:
    root = _parse_urdf(paths)
    urdf_dir = paths.out_urdf.parent
    missing: list[str] = []
    for mesh in root.iter("mesh"):
        filename = mesh.attrib.get("filename")
        if not filename:
            continue
        if filename.startswith(("package://", "http://", "https://")):
            continue
        candidate = filename if filename.startswith("/") else str(urdf_dir / filename)
        if not Path(candidate).is_file():
            missing.append(filename)
    if not missing:
        return ""
    shown = missing[:3]
    suffix = "…" if len(missing) > 3 else ""
    return f"{len(missing)} mesh path(s) unresolved: {shown}{suffix}"


def _check_single_tree(paths) -> str:
    """URDF should be a single connected kinematic tree (no orphan links)."""
    root = _parse_urdf(paths)
    links = {l.attrib.get("name", "") for l in root.findall("link")}
    children_of: dict[str, list[str]] = {n: [] for n in links}
    parents: dict[str, str] = {}
    for j in root.findall("joint"):
        p = j.find("parent")
        c = j.find("child")
        if p is None or c is None:
            continue
        pl = p.attrib.get("link", "")
        cl = c.attrib.get("link", "")
        if pl in links and cl in links:
            children_of.setdefault(pl, []).append(cl)
            parents[cl] = pl
    roots = [l for l in links if l not in parents]
    if len(roots) != 1:
        return f"expected single root, found {len(roots)}: {roots[:5]}"
    # Every link should be reachable from the single root.
    visited: set[str] = set()
    stack = [roots[0]]
    while stack:
        cur = stack.pop()
        if cur in visited:
            continue
        visited.add(cur)
        stack.extend(children_of.get(cur, []))
    orphans = links - visited
    if orphans:
        shown = sorted(orphans)[:5]
        return f"{len(orphans)} orphan link(s): {shown}"
    return ""


def _check_drift(paths) -> str:
    result = compose(paths)
    errs = check_drift(paths, result)
    if errs:
        return "; ".join(errs)
    return ""


# --------------------------- MJCF checks (#10–12) ------------------------- #


def _urdf_root_link(urdf_path: Path) -> str:
    """The URDF's actual root link (the one with no inbound joint)."""
    root = ET.parse(str(urdf_path)).getroot()
    links = {l.attrib.get("name", "") for l in root.findall("link")}
    children = {
        j.find("child").attrib.get("link", "")
        for j in root.findall("joint")
        if j.find("child") is not None
    }
    roots = [l for l in links if l and l not in children]
    if len(roots) != 1:
        raise SchemaError(
            f"{urdf_path}: expected single URDF root, found {roots}"
        )
    return roots[0]


def _check_mjcf_loads(paths) -> str:
    if mujoco is None:
        return "mujoco not installed"
    if not paths.out_mjcf.is_file():
        return f"missing {paths.out_mjcf}"
    errs = io.StringIO()
    try:
        with contextlib.redirect_stderr(errs):
            mujoco.MjModel.from_xml_path(str(paths.out_mjcf))
    except Exception as e:
        return f"load error: {e}"
    warn = errs.getvalue().strip()
    if warn:
        return f"warnings: {warn}"
    return ""


def _check_mjcf_actuator_order(paths, manifest: dict) -> str:
    """Actuator transmission joints, in MuJoCo's compiled order, must match
    manifest.joints[role] concatenated in role declaration order. This is
    what runtime controllers rely on to slice ctrl[] per-role.
    """
    if mujoco is None:
        return "mujoco not installed"
    model = mujoco.MjModel.from_xml_path(str(paths.out_mjcf))
    model_act_joints = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, model.actuator_trnid[i, 0])
        for i in range(model.nu)
    ]
    expected: list[str] = []
    for role, jlist in (manifest.get("joints") or {}).items():
        expected.extend(jlist)
    if model_act_joints != expected:
        return f"order mismatch: expected {expected}, got {model_act_joints}"
    return ""


def _check_mjcf_self_contact(paths) -> str:
    """Active contacts at qpos=0 clamp joints via friction (same failure
    mode as the per-component SELF_CONTACT check in §9 #8). The composer
    auto-emits mount-seam excludes for ancestor-descendant component
    pairs; this check catches anything missed.
    """
    if mujoco is None:
        return "mujoco not installed"
    model = mujoco.MjModel.from_xml_path(str(paths.out_mjcf))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    if data.ncon == 0:
        return ""
    pairs = set()
    for i in range(data.ncon):
        c = data.contact[i]
        b1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                               model.geom_bodyid[c.geom1])
        b2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                               model.geom_bodyid[c.geom2])
        pairs.add(tuple(sorted([b1, b2])))
    listing = "; ".join(f"{a} <-> {b}" for a, b in sorted(pairs))
    return f"{data.ncon} contacts at qpos=0 between {len(pairs)} pair(s): {listing}"


def _check_mjcf_frame_parity(paths, manifest: dict) -> str:
    """At qpos=0, every manifest.frames[role:frame] site in the MJCF must
    sit on the corresponding URDF link's world frame to 1e-5 m / 1e-5 rad.

    This is the cross-sim correctness check — catches axis-sign, euler-
    sequence, and origin-transcription bugs that per-component validators
    can miss at the workstation seam (mount xyz/rpy applied differently).
    """
    if mujoco is None:
        return "mujoco not installed"

    model = mujoco.MjModel.from_xml_path(str(paths.out_mjcf))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    urdf_root = _urdf_root_link(paths.out_urdf)
    urdf_world = urdf_link_world_poses(paths.out_urdf, urdf_root)

    POS_TOL = 1e-5
    ROT_TOL = 1e-5
    problems: list[str] = []
    checked = 0
    for key, urdf_link in (manifest.get("frames") or {}).items():
        site_name = key.replace(":", "_")
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if sid < 0:
            problems.append(f"{key}: site '{site_name}' not in MJCF")
            continue
        if urdf_link not in urdf_world:
            # Link sits inside a placeholder branch; skip rather than fail.
            continue
        urdf_pos, urdf_R = urdf_world[urdf_link]
        mjcf_pos = np.array(data.site_xpos[sid])
        mjcf_R = np.array(data.site_xmat[sid]).reshape(3, 3)
        pos_err = float(np.max(np.abs(mjcf_pos - urdf_pos)))
        rot_err = float(np.max(np.abs(mjcf_R - urdf_R)))
        if pos_err > POS_TOL:
            problems.append(f"{key}: pos_err={pos_err:.3e}")
        elif rot_err > ROT_TOL:
            problems.append(f"{key}: rot_err={rot_err:.3e}")
        checked += 1
    if not problems and checked == 0:
        return "no frames matched between manifest and URDF tree"
    if problems:
        head = "; ".join(problems[:5])
        suffix = f" (+{len(problems) - 5} more)" if len(problems) > 5 else ""
        return head + suffix
    return ""


# --------------------------- Driver --------------------------------------- #


def validate(workstation_dir: Path, assets_root: Path | None) -> int:
    try:
        paths = resolve_paths(workstation_dir, assets_root)
    except SchemaError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    # Pre-flight: manifest + URDF must exist before further checks make sense.
    for label, fn in [
        ("manifest_exists", lambda: _check_manifest_exists(paths)),
        ("urdf_exists", lambda: _check_urdf_exists(paths)),
    ]:
        msg = fn()
        if msg:
            print(f"FAIL {label}: {msg}", file=sys.stderr)
            return 1

    with paths.out_manifest.open() as f:
        manifest = yaml.safe_load(f)
    recipe = Recipe.load(paths.recipe)

    checks: list[tuple[str, Callable[[], str]]] = [
        ("manifest.recipe_sha256", lambda: _check_recipe_hash(paths, manifest)),
        ("manifest.component_hashes", lambda: _check_component_hashes(paths, recipe, manifest)),
        ("manifest.urdf_sha256", lambda: _check_urdf_hash(paths, manifest)),
        ("urdf.joint_counts", lambda: _check_joint_counts(paths, manifest)),
        ("urdf.ee_and_mount_links", lambda: _check_ee_and_mounts(paths, manifest)),
        ("urdf.mesh_resolution", lambda: _check_mesh_resolution(paths)),
        ("urdf.single_tree", lambda: _check_single_tree(paths)),
        ("composer.drift", lambda: _check_drift(paths)),
    ]
    # MJCF checks gated on a committed workstation.mjcf — they're skipped
    # for components that haven't authored their MJCFs yet.
    if paths.out_mjcf.is_file():
        checks.extend([
            ("mjcf.loads", lambda: _check_mjcf_loads(paths)),
            ("mjcf.actuator_order", lambda: _check_mjcf_actuator_order(paths, manifest)),
            ("mjcf.self_contact", lambda: _check_mjcf_self_contact(paths)),
            ("mjcf.frame_parity", lambda: _check_mjcf_frame_parity(paths, manifest)),
        ])
    results = _run(checks)

    ws_name = paths.workstation_dir.name
    any_fail = False
    for r in results:
        marker = "OK  " if r.ok else "FAIL"
        line = f"{marker} {ws_name}: {r.name}"
        if r.detail:
            line += f" — {r.detail}"
        print(line, file=sys.stderr if not r.ok else sys.stdout)
        if not r.ok:
            any_fail = True

    return 1 if any_fail else 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("workstation_dir", type=Path, help="path to workstation directory")
    p.add_argument("--assets-root", type=Path, default=None)
    args = p.parse_args(argv)
    return validate(args.workstation_dir, args.assets_root)


if __name__ == "__main__":
    raise SystemExit(main())
