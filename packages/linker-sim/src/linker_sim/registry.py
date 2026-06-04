"""Workstation registry.

Runtime entry point for loading composed workstation artifacts. Trusts
the committed `manifest.yaml` as the source of truth — schema validation
is the composer's job; the registry is read-only and sim-agnostic.

Typical use (from a sim backend):

    from linker_sim.registry import discover, load
    names = discover()                   # ["ar5_l6_bench_bimanual", "lkls73_i1_bimanual"]
    handle = load("ar5_l6_bench_bimanual")
    isaac_cfg = to_articulation_cfg(handle)   # backend-specific, see sim/backends/isaac

A `WorkstationHandle` carries everything a backend needs to construct its
native asset (URDF path, MJCF path, joint lists, frames, gains) without
reading the manifest a second time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# Resolved against the repo root by default. Overridable per-call for tests
# or when running from a different checkout.
# Layout: repo_root / packages / linker-sim / src / linker_sim / registry.py
# parents[4] climbs back to repo_root so we can reach `assets/workstations`.
# TODO(phase-2): swap for `linker_robot_assets.asset_root() / "workstations"`
# once the assets package is populated. Tracking: docs/REFACTOR_PLAN.md 2.5.
_DEFAULT_ROOT = Path(__file__).resolve().parents[4] / "assets" / "workstations"


# ----------------------------- Handle ------------------------------------- #


@dataclass(frozen=True)
class Gains:
    stiffness: float
    damping: float


@dataclass(frozen=True)
class WorkstationHandle:
    """Read-only view of a composed workstation.

    Fields mirror the manifest. Paths are absolute and pre-resolved so
    backend code can pass them straight to native loaders.
    """

    name: str
    dir: Path                          # workstations/<name>/
    urdf_path: Path                    # absolute
    mjcf_path: Path | None             # absolute; None when MJCFs not yet authored
    manifest_path: Path                # absolute
    recipe_path: Path                  # absolute

    # Kinematic summary (prefixed names; already role-namespaced by composer).
    joints: dict[str, list[str]]       # role -> ordered actuated joints
    mimic_joints: dict[str, list[str]] # role -> ordered mimic joints
    frames: dict[str, str]             # "role:frame" -> prefixed link name
    ee_link: str                       # prefixed — first arm-role's ee (back-compat)
    ee_links: dict[str, str]           # role -> prefixed ee link (one per arm role)
    base_link: str                     # prefixed

    # Merged component + recipe overrides.
    default_gains: dict[str, Gains]    # role -> gains
    gain_profiles: dict[str, dict[str, Gains]]  # role -> profile_name -> gains

    # Per-role XRDF paths for cuMotion collision spheres. Empty dict when
    # no component ships an XRDF.
    xrdf_paths: dict[str, Path]        # role -> absolute path to .xrdf

    # Provenance — useful for debugging which component/variant produced
    # which prefix.
    components: dict[str, "ComponentRef"]

    # Raw manifest dict for fields we haven't promoted to typed attrs yet.
    raw_manifest: dict[str, Any] = field(repr=False)

    def joint_names(self) -> list[str]:
        """All actuated joints across every role, in role-insertion order."""
        out: list[str] = []
        for js in self.joints.values():
            out.extend(js)
        return out

    def role_of(self, joint_name: str) -> str | None:
        for role, js in self.joints.items():
            if joint_name in js:
                return role
        return None


@dataclass(frozen=True)
class ComponentRef:
    name: str            # "arms/ar5"
    variant: str | None  # "left" | "right" | "default" | None
    sha256: str


# ----------------------------- Loading ------------------------------------ #


class RegistryError(ValueError):
    """Raised on missing / malformed workstation artifacts."""


def discover(root: Path | None = None) -> list[str]:
    """List workstations that have a `recipe.yaml` under `root`.

    Does not require a composed `workstation.urdf` — discovery reports the
    set of authored workstations; a workstation without a composed URDF
    is visible but will fail to `load`.
    """
    root = (root or _DEFAULT_ROOT).resolve()
    if not root.is_dir():
        return []
    return sorted(
        p.name
        for p in root.iterdir()
        if p.is_dir() and (p / "recipe.yaml").is_file()
    )


def load(name: str, root: Path | None = None) -> WorkstationHandle:
    """Load a workstation by name.

    Raises `RegistryError` if the workstation directory doesn't exist, if
    its `manifest.yaml` is missing (composer hasn't been run), or if the
    manifest's declared URDF artifact is missing.
    """
    root = (root or _DEFAULT_ROOT).resolve()
    ws_dir = root / name
    if not ws_dir.is_dir():
        raise RegistryError(
            f"workstation {name!r} not found under {root} "
            f"(available: {discover(root)})"
        )
    manifest_path = ws_dir / "manifest.yaml"
    recipe_path = ws_dir / "recipe.yaml"
    if not manifest_path.is_file():
        raise RegistryError(
            f"{name}: manifest.yaml missing — run "
            f"`python -m linker_sim.tools.composer.compose assets/workstations/{name}`"
        )

    with manifest_path.open() as f:
        m = yaml.safe_load(f) or {}
    if not isinstance(m, dict):
        raise RegistryError(f"{manifest_path}: manifest root is not a mapping")

    artifacts = m.get("artifacts", {}) or {}
    urdf_rel = artifacts.get("urdf")
    if not urdf_rel:
        raise RegistryError(f"{manifest_path}: artifacts.urdf is empty")
    urdf_path = (ws_dir / urdf_rel).resolve()
    if not urdf_path.is_file():
        raise RegistryError(
            f"{name}: declared URDF {urdf_path} missing — re-run composer"
        )

    mjcf_rel = artifacts.get("mjcf")
    mjcf_path: Path | None = None
    if mjcf_rel:
        candidate = (ws_dir / mjcf_rel).resolve()
        if candidate.is_file():
            mjcf_path = candidate
        # If the manifest declares an MJCF but the file is missing, we
        # treat it as "MJCF not available" rather than an error so that
        # Isaac-only workflows can proceed while PR #1b is pending.

    return WorkstationHandle(
        name=m.get("name", name),
        dir=ws_dir,
        urdf_path=urdf_path,
        mjcf_path=mjcf_path,
        manifest_path=manifest_path,
        recipe_path=recipe_path,
        joints={k: list(v) for k, v in (m.get("joints") or {}).items()},
        mimic_joints={k: list(v) for k, v in (m.get("mimic_joints") or {}).items()},
        frames=dict(m.get("frames") or {}),
        ee_link=str(m.get("ee_link") or ""),
        ee_links={str(role): str(link) for role, link in (m.get("ee_links") or {}).items()},
        base_link=str(m.get("base_link") or ""),
        default_gains={
            role: Gains(
                stiffness=float(g["stiffness"]),
                damping=float(g["damping"]),
            )
            for role, g in (m.get("default_gains") or {}).items()
        },
        gain_profiles={
            role: {
                pname: Gains(
                    stiffness=float(g["stiffness"]),
                    damping=float(g["damping"]),
                )
                for pname, g in profiles.items()
            }
            for role, profiles in (m.get("gain_profiles") or {}).items()
        },
        xrdf_paths={
            str(role): (ws_dir / rel).resolve()
            for role, rel in (m.get("xrdf_paths") or {}).items()
            if (ws_dir / rel).resolve().is_file()
        },
        components={
            role: ComponentRef(
                name=str(p["name"]),
                variant=(str(p["variant"]) if p.get("variant") else None),
                sha256=str(p["sha256"]),
            )
            for role, p in (m.get("components") or {}).items()
        },
        raw_manifest=m,
    )
