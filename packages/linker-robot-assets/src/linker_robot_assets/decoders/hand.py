"""Hand telemetry decoder — linear-fit placeholder.

WARNING: SDK-pending. The Linker SDK has not yet defined an angle
convention; this module ships a linear interpolation from an SDK-shaped
0–100 value per channel to the URDF [lower, upper] limit of the
corresponding actuated joint. When the SDK lands an angle convention,
bump `CONVENTION` to `sdk-vN` and re-run any bagged data stamped with
`linear-fit-v0`.

Convention (verified empirically against Linker Hand O6 telemetry):

    sdk_value = 100 -> joint at URDF lower limit (rest / open)
    sdk_value = 0   -> joint at URDF upper limit (full travel)
    joint = lower + (100 - sdk)/100 * (upper - lower)

Matches `linker_sim.io.replay.hands` (raw=full-scale byte → lower limit)
but operates on the SDK 0–100 percent scale rather than 0–255 bytes.

Tracked at:

- docs/known_limitations.md (linear-fit + UMI-Dex path hack)
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml

from linker_robot_assets import asset_root


CONVENTION = "linear-fit-v0"

_VALID_SIDES = ("left", "right")
_SIDE_TO_PREFIX = {"left": "l", "right": "r"}


def _resolve_component_dir(name: str, component_root: Path | None) -> Path:
    root = component_root or (asset_root() / "components" / "hands")
    cdir = root / name
    if not cdir.is_dir():
        raise FileNotFoundError(
            f"hand component {name!r} not found at {cdir} "
            f"(component_root={root})"
        )
    return cdir


def _read_decoder_yaml(cdir: Path) -> dict:
    path = cdir / "decoder.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing — author it per Phase 4.2")
    with path.open() as f:
        spec = yaml.safe_load(f) or {}
    declared = spec.get("convention")
    if declared != CONVENTION:
        raise ValueError(
            f"{path}: convention {declared!r} != module CONVENTION {CONVENTION!r}. "
            "Bump the file or the module to match."
        )
    channels = spec.get("channels") or []
    if not channels:
        raise ValueError(f"{path}: empty 'channels' list")
    return spec


def _expand_template(name: str, side: str) -> str:
    return name.replace("{S}", _SIDE_TO_PREFIX[side])


def _read_urdf_limits(
    urdf_path: Path, joint_names: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Extract per-joint (lower, upper) from a URDF via xml.etree.

    Avoids pulling yourdfpy as a dep — joint limits are a single xpath
    query and the rest of yourdfpy's machinery is unused here.
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    by_name = {j.get("name"): j for j in root.findall("joint")}
    lo = np.zeros(len(joint_names), dtype=np.float64)
    hi = np.zeros(len(joint_names), dtype=np.float64)
    for i, name in enumerate(joint_names):
        joint = by_name.get(name)
        if joint is None:
            raise KeyError(
                f"{urdf_path}: joint {name!r} not found "
                f"(available: {sorted(by_name)})"
            )
        limit = joint.find("limit")
        if limit is None:
            raise ValueError(
                f"{urdf_path}: joint {name!r} has no <limit> element"
            )
        lo[i] = float(limit.get("lower", "0"))
        hi[i] = float(limit.get("upper", "0"))
    return lo, hi


def _urdf_actuated_order(urdf_path: Path) -> list[str]:
    """Actuated (non-fixed, non-mimic) joint names, in URDF document order.

    Mirrors ``composer.urdf_ops.collect_joint_names`` so the decoder's output
    column order lines up with ``handle.joints[role]`` — the order the replay
    pipeline feeds hand columns against positionally.
    """
    root = ET.parse(urdf_path).getroot()
    names: list[str] = []
    for j in root.findall("joint"):
        if j.get("type") == "fixed" or j.find("mimic") is not None:
            continue
        name = j.get("name")
        if name:
            names.append(name)
    return names


def _resolve_hand_urdf(cdir: Path, name: str, side: str) -> Path:
    """Locate a hand's per-side URDF, tolerating both component layouts.

    Most hands (l6/l25/l20lite/l30) use ``variants/<side>/hand.urdf``; o6 ships
    a flat ``<name>_<side>.urdf`` at the component root. Try the variants path
    first, then fall back to the flat one.
    """
    candidates = (
        cdir / "variants" / side / "hand.urdf",
        cdir / f"{name}_{side}.urdf",
    )
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(
        f"{name}/{side}: no hand URDF found "
        f"(tried {[str(c) for c in candidates]})"
    )


def decode_hand(
    name: str,
    side: str,
    sdk_0_100: np.ndarray,
    *,
    legacy: bool = False,
    component_root: Path | None = None,
) -> np.ndarray:
    """Linear interp from SDK [0, 100] to URDF [lower, upper] per joint.

    Args:
        name: hand component directory name (e.g. ``"linkerhand_l6"``).
        side: ``"left"`` or ``"right"``.
        sdk_0_100: per-channel SDK values, shape ``(n_channels,)`` or
            ``(T, n_channels)``. Float; values outside [0, 100] are clipped.
            Columns are in the selected channel list's (SDK) order.
        legacy: use ``decoder.yaml::legacy_channels`` (an alternate SDK
            channel order shipped by early/buggy client devices) instead of
            the default ``channels``. Both list the same joints; only the
            input column order differs.
        component_root: override the asset-tree component root (test-only).

    Returns:
        Same shape as ``sdk_0_100``, dtype float32, in radians, reordered
        from SDK channel order into the URDF actuated-joint document order
        (== ``handle.joints[role]``, which replay feeds positionally).

    Raises:
        FileNotFoundError: component dir or ``decoder.yaml`` missing.
        ValueError: ``decoder.yaml`` convention mismatch, channel count
            mismatch, missing ``legacy_channels`` when ``legacy=True``, or a
            joint without a ``<limit>`` element.
        KeyError: a templated joint name not present in the variant URDF.
    """
    if side not in _VALID_SIDES:
        raise ValueError(f"side {side!r} not in {_VALID_SIDES}")

    cdir = _resolve_component_dir(name, component_root)
    spec = _read_decoder_yaml(cdir)
    key = "legacy_channels" if legacy else "channels"
    chans = spec.get(key)
    if not chans:
        raise ValueError(
            f"{cdir / 'decoder.yaml'}: no {key!r} list "
            f"(required for legacy={legacy})"
        )
    joint_names = [_expand_template(c, side) for c in chans]

    sdk = np.asarray(sdk_0_100, dtype=np.float32)
    if sdk.shape[-1] != len(joint_names):
        raise ValueError(
            f"{name}/{side}: decoder.yaml has {len(joint_names)} channels "
            f"but input has shape {sdk.shape} (last dim should be "
            f"{len(joint_names)})"
        )

    urdf_path = _resolve_hand_urdf(cdir, name, side)
    lo, hi = _read_urdf_limits(urdf_path, joint_names)

    # Optional per-joint clip overrides (joint name -> [lo, hi]).
    overrides = spec.get("clip_overrides") or {}
    for jname, bounds in overrides.items():
        try:
            i = joint_names.index(_expand_template(jname, side))
        except ValueError:
            # Override may be templated/untemplated; try the other form.
            i = joint_names.index(jname) if jname in joint_names else -1
        if i >= 0:
            lo[i] = float(bounds[0])
            hi[i] = float(bounds[1])

    sdk_clipped = np.clip(sdk, 0.0, 100.0)
    out = lo.astype(np.float32) + ((100.0 - sdk_clipped) / 100.0) * (hi - lo).astype(np.float32)

    # `channels` is in SDK/hardware order; the sim feeds hand columns
    # positionally against the URDF's actuated-joint document order
    # (== handle.joints[role]). Reorder so returned columns line up with it.
    manifest = _urdf_actuated_order(urdf_path)
    if sorted(manifest) != sorted(joint_names):
        raise ValueError(
            f"{name}/{side}: decoder.yaml channels {joint_names} do not match "
            f"the URDF's actuated joints {manifest} ({urdf_path})"
        )
    channel_idx = {jn: i for i, jn in enumerate(joint_names)}
    perm = [channel_idx[m] for m in manifest]
    return out[..., perm].astype(np.float32)
