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

from sim.io.replay import hands as hand_decoders


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
        data = np.load(npz, allow_pickle=False)
        if self.field not in data.files:
            raise KeyError(
                f"field {self.field!r} not in {npz} (keys: {list(data.files)})"
            )
        arr = data[self.field]
        if arr.ndim != 2:
            raise ValueError(f"expected 2-D telemetry array, got {arr.shape}")
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
