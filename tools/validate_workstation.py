#!/usr/bin/env python3
"""Validate a workstation's composed artifacts.

Checks performed (PR #1 scope):
  1. Manifest self-consistency: recipe/component hashes match on-disk sources,
     committed workstation.urdf matches manifest.artifacts.urdf_sha256.
  2. Kinematic structure: URDF has expected actuated-joint counts per role;
     declared EE link and every mount frame resolve to existing links.
  3. Mesh resolution: every `<mesh filename=>` path resolves on disk.
  4. Drift: re-run composer in memory; committed artifacts match fresh output.

Deferred to PR #1b:
  - MJCF loading (requires hand-authored component MJCFs).
  - Cross-sim joint-order consistency (URDF vs MJCF).
  - Gravity-compensated hold-drift on MuJoCo.

Exit codes:
  0 — all checks pass
  1 — one or more checks failed
  2 — usage / structural error
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

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
