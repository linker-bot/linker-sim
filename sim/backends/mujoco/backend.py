"""MuJoCo `SimBackend` stub.

Raises on construction with the message that unblocks the caller
(`PR #1b MJCFs`). Kept to validate the Protocol in
`sim/backends/base.py` against a second target.
"""

from __future__ import annotations

from dataclasses import dataclass, field


_UNBLOCK_MSG = (
    "MuJoCo backend is not yet implemented. Blocked on PR #1b "
    "(component MJCF authoring: see docs/component_mjcf_authoring.md). "
    "Once `assets/components/*/variants/*/arm.mjcf` etc. exist and "
    "`tools/composer/mjcf_ops.compose_mjcf` is fleshed out, a `workstation.mjcf` "
    "will be available for `mujoco.MjModel.from_xml_path(...)` to consume."
)


@dataclass
class MujocoBackendCfg:
    workstations: dict[str, str] = field(default_factory=lambda: {"robot": "ar5_l6_bench"})
    num_envs: int = 1
    dt: float = 1.0 / 500.0
    device: str = "cpu"


class MujocoSimBackend:
    def __init__(self, cfg: MujocoBackendCfg):
        raise NotImplementedError(_UNBLOCK_MSG)

    def step(self) -> None: raise NotImplementedError(_UNBLOCK_MSG)
    def write_data(self) -> None: raise NotImplementedError(_UNBLOCK_MSG)
    def reset(self, env_ids=None) -> None: raise NotImplementedError(_UNBLOCK_MSG)
    def close(self) -> None: raise NotImplementedError(_UNBLOCK_MSG)
