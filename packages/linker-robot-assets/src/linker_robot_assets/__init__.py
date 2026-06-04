"""linker-robot-assets — bundled robot asset tree + composer + validators.

This package ships:

- the asset tree at ``linker_robot_assets/assets/`` (URDF / MJCF / XRDF +
  meshes, organised by ``components/`` and ``workstations/``);
- the composer that builds workstation URDFs from components
  (``linker_robot_assets.composer``);
- two validators (``linker_robot_assets.validate_workstation`` and
  ``linker_robot_assets.validate_component_mjcf``);
- a loader API exposed at the package top level for downstream consumers
  (``linker-sim``'s registry, real-robot teleop tools, etc.).

The composer + validators are gated behind the ``[authoring]`` extra so
that runtime consumers (which only need ``asset_root`` /
``load_manifest`` / ``workstations``) get a minimal install.
"""

from __future__ import annotations

from pathlib import Path

import yaml

# Assets ship inside the package (sibling of __init__.py) so editable
# installs and built wheels resolve identically. Layered alongside the
# Python package, not under a parallel `assets/` at the package root —
# this avoids the hatch force-include + editable-install path-drift
# problem.
_ASSET_ROOT = Path(__file__).resolve().parent / "assets"

__all__ = ["asset_root", "workstations", "load_manifest"]


def asset_root() -> Path:
    """Return the on-disk root of the bundled asset tree.

    Layout under the returned path::

        components/{arms,bases,hands}/<name>/{meta.yaml,variants/...}
        workstations/<name>/{recipe.yaml, manifest.yaml,
                             workstation.urdf, workstation.mjcf}
    """
    return _ASSET_ROOT


def workstations() -> list[str]:
    """List workstation names that are composed (have ``manifest.yaml``).

    Returns the directory basenames sorted alphabetically. Workstations
    without a committed ``manifest.yaml`` are excluded — those need
    ``python -m linker_robot_assets.composer.compose`` to be runnable.
    """
    ws_dir = _ASSET_ROOT / "workstations"
    if not ws_dir.is_dir():
        return []
    return sorted(
        p.name
        for p in ws_dir.iterdir()
        if p.is_dir() and (p / "manifest.yaml").is_file()
    )


def load_manifest(name: str) -> dict:
    """Load a workstation's ``manifest.yaml`` and return it as a dict.

    Raises ``FileNotFoundError`` if the workstation directory or manifest
    is missing — call ``workstations()`` first to enumerate composed names.
    """
    manifest_path = _ASSET_ROOT / "workstations" / name / "manifest.yaml"
    with manifest_path.open() as f:
        return yaml.safe_load(f)
