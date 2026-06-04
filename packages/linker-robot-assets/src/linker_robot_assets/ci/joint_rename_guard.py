"""Refuse minor/patch version bumps when a workstation's joint names change.

Run as ``python -m linker_robot_assets.ci.joint_rename_guard`` (typically
invoked by ``check_drift.sh`` when ``CHECK_JOINT_RENAMES=1`` is set).

For each workstation present in BOTH the current tree and the previous git
tag, compare the actuated + mimic joint name sets across all roles. If
the sets differ (rename, removal, or addition within an existing
workstation), the package version must bump its **major** component,
since downstream consumers loading by joint name will break otherwise.

Exits 0 when:
- there is no previous tag (first release), or
- every common workstation has identical joint sets, or
- joint sets changed AND the major version bumped.

Exits 1 when joint sets changed and only minor/patch bumped.
Exits 0 with a warning when the previous tag predates the uv-workspace
layout (its pyproject.toml lives at the repo root, not in
``packages/linker-sim/``).
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import yaml

from linker_robot_assets import asset_root

# Pyproject path — this is where the linker-sim version lives. The
# workspace itself doesn't carry a [project] table.
_PYPROJECT_REL = "packages/linker-sim/pyproject.toml"


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _previous_tag() -> str | None:
    """Most recent ancestor tag, or None if there are no tags."""
    try:
        return _git("describe", "--tags", "--abbrev=0", "HEAD^")
    except subprocess.CalledProcessError:
        try:
            return _git("describe", "--tags", "--abbrev=0")
        except subprocess.CalledProcessError:
            return None


def _read_version(text: str) -> tuple[int, int, int] | None:
    """Pull `version = "X.Y.Z"` out of a pyproject.toml string."""
    m = re.search(r'^version\s*=\s*"(\d+)\.(\d+)\.(\d+)"', text, flags=re.M)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def _joint_set_from_manifest(manifest: dict) -> frozenset[str]:
    """Flatten {role: [joint, …]} for actuated + mimic into a single set."""
    out: set[str] = set()
    for joints in (manifest.get("joints") or {}).values():
        out.update(joints)
    for joints in (manifest.get("mimic_joints") or {}).values():
        out.update(joints)
    return frozenset(out)


def _current_joint_sets() -> dict[str, frozenset[str]]:
    """Read joint sets per workstation from the current asset tree."""
    ws_root = asset_root() / "workstations"
    out: dict[str, frozenset[str]] = {}
    for ws_dir in sorted(ws_root.iterdir()):
        manifest = ws_dir / "manifest.yaml"
        if not manifest.is_file():
            continue
        with manifest.open() as f:
            out[ws_dir.name] = _joint_set_from_manifest(yaml.safe_load(f) or {})
    return out


def _tagged_joint_sets(tag: str) -> dict[str, frozenset[str]] | None:
    """Joint sets at ``tag``. None if the tag predates the new layout."""
    # Find every committed manifest.yaml at the tag, regardless of where
    # `assets/` lived back then. `git ls-tree` walks the tree at the tag.
    try:
        ls = _git("ls-tree", "-r", "--name-only", tag)
    except subprocess.CalledProcessError:
        return None
    manifests = [p for p in ls.splitlines() if p.endswith("/manifest.yaml") and "workstations/" in p]
    if not manifests:
        return None
    out: dict[str, frozenset[str]] = {}
    for path in manifests:
        # workstation name = parent dir name
        name = Path(path).parent.name
        try:
            text = _git("show", f"{tag}:{path}")
        except subprocess.CalledProcessError:
            continue
        out[name] = _joint_set_from_manifest(yaml.safe_load(text) or {})
    return out or None


def main() -> int:
    tag = _previous_tag()
    if tag is None:
        print("[joint-rename-guard] no previous tag — skipping (first release)")
        return 0

    tagged = _tagged_joint_sets(tag)
    if tagged is None:
        print(
            f"[joint-rename-guard] tag {tag} has no committed manifests — "
            "skipping (likely predates the composer pipeline)"
        )
        return 0

    current = _current_joint_sets()

    common = sorted(set(current) & set(tagged))
    changed: list[str] = [ws for ws in common if current[ws] != tagged[ws]]
    if not changed:
        print(f"[joint-rename-guard] all {len(common)} workstation(s) stable vs {tag}")
        return 0

    # Joint sets shifted. Read both versions.
    current_pyproject = (Path(__file__).resolve().parents[4] / _PYPROJECT_REL).read_text()
    cur_v = _read_version(current_pyproject)
    try:
        tag_pyproject = _git("show", f"{tag}:{_PYPROJECT_REL}")
    except subprocess.CalledProcessError:
        # Tag predates the workspace split: pyproject was at repo root.
        try:
            tag_pyproject = _git("show", f"{tag}:pyproject.toml")
        except subprocess.CalledProcessError:
            print(
                f"[joint-rename-guard] cannot resolve tag pyproject for {tag}; "
                "skipping",
                file=sys.stderr,
            )
            return 0
    tag_v = _read_version(tag_pyproject)

    if cur_v is None or tag_v is None:
        print(
            "[joint-rename-guard] could not parse versions — skipping check",
            file=sys.stderr,
        )
        return 0

    bumped_major = cur_v[0] > tag_v[0]
    print(
        f"[joint-rename-guard] joint sets changed in {len(changed)} workstation(s) "
        f"vs {tag}: {', '.join(changed)}"
    )
    print(f"[joint-rename-guard] tag version {'.'.join(map(str, tag_v))} → "
          f"current {'.'.join(map(str, cur_v))}")

    for ws in changed:
        added = current[ws] - tagged[ws]
        removed = tagged[ws] - current[ws]
        if added:
            print(f"  {ws}: + {sorted(added)}")
        if removed:
            print(f"  {ws}: - {sorted(removed)}")

    if bumped_major:
        print("[joint-rename-guard] major version bumped — OK")
        return 0

    print(
        "[joint-rename-guard] joint names changed without a major bump. "
        "Bump the major version in packages/linker-sim/pyproject.toml or "
        "revert the joint rename.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
