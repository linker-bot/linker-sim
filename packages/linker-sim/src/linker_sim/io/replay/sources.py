"""Replay sources.

A source is a (role -> per-frame joint-target table) container plus a
sample rate. `TelemetryNpzSource` is the only concrete impl right now;
it reads a numpy `.npz` (or directory containing `telemetry.npz`) and
slices a flat (T, N_total) column block into per-role (T, n_joints)
arrays according to a YAML-driven layout.

The layout describes how a recording's column ordering maps onto the
workstation's roles, including:
- column slice [start, end)
- a sign flip if the recording's joint-axis convention differs
- an optional `decoder` (e.g. "linker_l6") for non-radian columns

Decoders are applied lazily via `bind_robot()` once the workstation's
actuated-joint limits are known.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import yaml

from linker_robot_assets import asset_root
from linker_sim.io.replay import hands as hand_decoders


_SIDE_TO_PREFIX = {"left": "l", "right": "r"}


@dataclass
class RoleLayout:
    """One role's slice of a flat telemetry column block."""

    cols: tuple[int, int]            # [start, end)
    sign: float = 1.0                # multiplied into raw values when no decoder
    decoder: str | None = None       # name in sim.io.replay.hands._DECODERS


class ReplaySource(Protocol):
    """Protocol every source implements."""

    hz: float
    num_frames: int
    roles: tuple[str, ...]

    def bind_robot(self, robot: Any) -> None: ...
    def joint_targets(self, t: int) -> dict[str, np.ndarray]: ...


@dataclass
class TelemetryNpzSource:
    """Real-robot telemetry stored as a flat (T, N) array per field.

    Args:
        path: directory containing `telemetry.npz` or the file itself.
        layout: role -> RoleLayout. Each role's `cols` slice must match
            the actuated-joint count for that role on the bound robot.
        field: which key to read out of the npz (e.g. "qpos" or "actions").
        hz: sample rate. Used to time-pace the replay loop.
    """

    path: str | Path
    layout: dict[str, RoleLayout | dict]
    field: str = "qpos"
    hz: float = 30.0

    def __post_init__(self):
        npz = Path(self.path)
        if npz.is_dir():
            npz = npz / "telemetry.npz"
        if not npz.is_file():
            raise FileNotFoundError(f"telemetry not found: {npz}")
        with np.load(npz, allow_pickle=False) as data:
            if self.field not in data.files:
                raise KeyError(
                    f"field {self.field!r} not in {npz} (keys: {list(data.files)})"
                )
            arr = np.asarray(data[self.field])
        if arr.ndim != 2:
            raise ValueError(f"expected 2-D telemetry array, got {arr.shape}")
        n_cols = int(arr.shape[1])
        self._npz_path = npz
        self._raw_full = arr
        self.num_frames = int(arr.shape[0])

        # Normalize layout: accept dicts (Hydra) or RoleLayout instances.
        norm: dict[str, RoleLayout] = {}
        for role, spec in self.layout.items():
            if isinstance(spec, RoleLayout):
                norm[role] = spec
            else:
                cols = tuple(spec["cols"])
                norm[role] = RoleLayout(
                    cols=(int(cols[0]), int(cols[1])),
                    sign=float(spec.get("sign", 1.0)),
                    decoder=spec.get("decoder"),
                )
        for role, spec in norm.items():
            lo, hi = spec.cols
            if not (0 <= lo < hi <= n_cols):
                raise ValueError(
                    f"role {role!r}: cols {spec.cols} out of range "
                    f"for telemetry with {n_cols} columns"
                )
        self.layout = norm
        self.roles = tuple(self.layout.keys())

        # Pre-slice raw per-role tables; decoded values fill in at bind_robot.
        self._raw_by_role: dict[str, np.ndarray] = {
            role: arr[:, spec.cols[0]:spec.cols[1]].astype(np.float32, copy=False)
            for role, spec in self.layout.items()
        }
        self._targets_by_role: dict[str, np.ndarray] = {}

    def bind_robot(self, robot: Any) -> None:
        """Resolve decoders into final radian-valued target tables.

        Must be called before `joint_targets`. Verifies that each role's
        column count matches the robot's actuated-joint count for that
        role and applies the configured decoder + sign.

        For roles with a hand decoder, raw columns are permuted from
        SDK-channel order (component's `decoder.yaml::channels`) into
        manifest joint order before decoding, so that the physical
        URDF/manifest joint order can differ from the hardware's data
        protocol. If a component has no `decoder.yaml`, the raw slice
        is fed positionally against manifest order (legacy behavior).
        """
        for role, spec in self.layout.items():
            ids = robot.actuated_joint_ids_of(role)
            n_expected = int(ids.numel() if hasattr(ids, "numel") else len(ids))
            raw = self._raw_by_role[role]
            if raw.shape[1] != n_expected:
                raise ValueError(
                    f"role {role!r}: source provides {raw.shape[1]} columns "
                    f"but robot has {n_expected} actuated joints"
                )
            if spec.decoder is None:
                self._targets_by_role[role] = (spec.sign * raw).astype(np.float32)
            else:
                perm = _channel_permutation(robot, role)
                if perm is not None:
                    raw = raw[:, perm]
                lo, hi = robot.actuated_joint_limits_of(role)
                lo_np = lo.detach().cpu().numpy() if hasattr(lo, "detach") else np.asarray(lo)
                hi_np = hi.detach().cpu().numpy() if hasattr(hi, "detach") else np.asarray(hi)
                decode = hand_decoders.get(spec.decoder)
                self._targets_by_role[role] = decode(raw, lo_np, hi_np).astype(np.float32)

    def joint_targets(self, t: int) -> dict[str, np.ndarray]:
        if not self._targets_by_role:
            raise RuntimeError("bind_robot() must be called before joint_targets()")
        t = max(0, min(int(t), self.num_frames - 1))
        return {role: self._targets_by_role[role][t] for role in self.roles}

    def describe(self) -> str:
        return (
            f"TelemetryNpzSource({self._npz_path.name}, field={self.field}, "
            f"hz={self.hz}, frames={self.num_frames}, "
            f"roles={list(self.roles)})"
        )


def _channel_permutation(robot: Any, role: str) -> np.ndarray | None:
    """Permutation from SDK channel order → manifest joint order for a role.

    Consults `<component>/decoder.yaml::channels`, expands `{S}` for the
    component variant, and prefixes with the role name to match manifest
    joint names. Returns `perm` such that `raw[:, perm]` reorders SDK-
    ordered columns into manifest-ordered columns; returns `None` when
    no reorder is needed (identity permutation, or no `decoder.yaml`
    to consult).

    Raises `ValueError` if any manifest joint has no matching SDK channel,
    or if the two lists differ in length.
    """
    handle = getattr(robot, "handle", None)
    if handle is None:
        return None
    comp_ref = handle.components.get(role) if handle.components else None
    if comp_ref is None:
        return None
    decoder_yaml = asset_root() / "components" / comp_ref.name / "decoder.yaml"
    if not decoder_yaml.is_file():
        return None
    with decoder_yaml.open() as f:
        dspec = yaml.safe_load(f) or {}
    channels = dspec.get("channels") or []
    if not channels:
        return None
    side = comp_ref.variant
    if side not in _SIDE_TO_PREFIX:
        raise ValueError(
            f"role {role!r}: decoder.yaml expects variant left/right, got {side!r}"
        )
    s_char = _SIDE_TO_PREFIX[side]
    sdk_names = [f"{role}_{c.replace('{S}', s_char)}" for c in channels]
    manifest_names = list(handle.joints.get(role) or [])
    if len(sdk_names) != len(manifest_names):
        raise ValueError(
            f"role {role!r}: decoder.yaml lists {len(sdk_names)} channels "
            f"but manifest has {len(manifest_names)} actuated joints "
            f"({decoder_yaml})"
        )
    sdk_idx = {name: i for i, name in enumerate(sdk_names)}
    perm = np.empty(len(manifest_names), dtype=np.int64)
    for j, mname in enumerate(manifest_names):
        k = sdk_idx.get(mname)
        if k is None:
            raise ValueError(
                f"role {role!r}: manifest joint {mname!r} not listed in "
                f"decoder.yaml channels {sdk_names} ({decoder_yaml})"
            )
        perm[j] = k
    if np.array_equal(perm, np.arange(len(perm), dtype=np.int64)):
        return None
    return perm
