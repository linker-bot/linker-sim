"""MJCF composition operations.

Structure parallels urdf_ops.py but operates on MuJoCo MJCF. MJCF and URDF
have fundamentally different element organization (MJCF groups meshes under
`<asset>`, nests joints inside `<body>`, puts drives under `<actuator>`,
and scopes `<default>` classes), so this is not a thin wrapper — it's its
own composer pass.

PR #1 scope: infrastructure + the "skip if any component MJCF is missing"
fast path. Full composition (the merge of <asset>/<worldbody>/<actuator>
/<equality>/<contact>/<sensor> with role-prefixed names and deduped mesh
assets) is deferred to PR #1b, which lands alongside the hand-authored
component MJCF files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .schemas import ComponentMeta, Variant


@dataclass
class MjcfAvailability:
    """Whether every component declares + ships an MJCF file."""

    all_present: bool
    missing: list[str]  # roles with no MJCF declared or file missing


def check_mjcf_availability(
    components_with_variant: list[tuple[str, ComponentMeta, Variant]],
) -> MjcfAvailability:
    """Inspect each component variant for a usable MJCF source.

    Returns `all_present=True` only if every role declares `mjcf` in its
    variant and the file exists on disk.
    """
    missing: list[str] = []
    for role, meta, variant in components_with_variant:
        if not variant.mjcf:
            missing.append(f"{role} ({meta.name}/{variant.name}): no mjcf declared")
            continue
        path = meta.source_dir / variant.mjcf
        if not path.is_file():
            missing.append(f"{role} ({meta.name}/{variant.name}): {path} not found")
    return MjcfAvailability(all_present=not missing, missing=missing)


def compose_mjcf(*args, **kwargs):
    """Compose a workstation MJCF.

    Planned for PR #1b. The algorithm mirrors URDF composition:

      1. Namespace body/joint/site/geom/actuator/sensor names with the role
         prefix; rename `<default>` classes to `<role>_<class>` and rewrite
         every `class="..."` attribute.
      2. Merge top-level sections under one `<mujoco>` root:
         `<asset>`, `<default>`, `<worldbody>`, `<actuator>`, `<sensor>`,
         `<contact>`, `<equality>`. Dedupe mesh assets by (file, scale).
      3. Reparent each component's root `<body>` under its mount parent
         with `<pos>`/`<quat>` from the recipe's mount xyz/rpy.
      4. If `freeze_base` is set, the base component's root body has no
         freejoint; otherwise no implicit constraint is added (the
         authored MJCF decides).
      5. Emit deterministic output via `determinism.serialize`.

    The edge-case handling (material/mesh dedup across components, default
    class scoping, <contact> exclude pairs at mount seams, sensor site
    remapping) is the detail work that belongs with a specific authored
    MJCF in hand to test against.
    """
    raise NotImplementedError(
        "MJCF composition lands in PR #1b alongside hand-authored component MJCFs"
    )
