"""Viser `Robot` implementation — replay-only.

Tracks an internal `joint_buffer` (1-D numpy array, length =
URDF actuated-joint count, ordered by `urdf.actuated_joint_names`).
Replay writes targets here via `set_joint_position_target`; the backend
reads `joint_buffer` in `write_data()` and pushes it to the Viser scene.

Implements the subset of `linker_sim.backends.base.Robot` that
`linker_sim.runtime.replay.run_replay()` and
`linker_sim.io.replay.sources.TelemetryNpzSource.bind_robot()` actually
call:

- `actuated_joint_ids_of(role)` / `joint_ids_of(role)`
- `actuated_joint_limits_of(role)` (for hand decoders)
- `set_joint_position_target(targets, joint_ids)`
- `write_joint_state(joint_pos, joint_vel, env_ids=None)`
- `joint_pos`, `joint_pos_default`, `joint_vel_default`

Methods related to dynamics (Jacobian, mass matrix, gravity), effort
control, and frame pose introspection raise NotImplementedError.
"""

from __future__ import annotations

import numpy as np
import torch

from linker_sim.registry import WorkstationHandle


_REPLAY_ONLY_MSG = (
    "Viser backend is replay-only. "
    "Use the MuJoCo or Isaac backend for dynamics / control."
)


class ViserRobot:
    """Viser implementation of the replay subset of `Robot`."""

    def __init__(self, handle: WorkstationHandle, urdf):
        self.handle = handle
        self._urdf = urdf
        self.num_envs = 1
        self.device = torch.device("cpu")

        actuated_names = list(urdf.actuated_joint_names)
        self._actuated_names = actuated_names
        name_to_idx = {n: i for i, n in enumerate(actuated_names)}
        self._n_actuated = len(actuated_names)

        self._actuated_ids_by_role: dict[str, torch.Tensor] = {}
        for role, role_joints in handle.joints.items():
            try:
                ids = [name_to_idx[name] for name in role_joints]
            except KeyError as exc:
                raise KeyError(
                    f"role {role!r}: actuated joint {exc.args[0]!r} from manifest "
                    f"not found in URDF actuated joints "
                    f"(available: {actuated_names})"
                ) from exc
            self._actuated_ids_by_role[role] = torch.tensor(
                ids, dtype=torch.long, device=self.device
            )

        # joint_ids_of() exposes "actuated + mimic"; mimics aren't in the
        # URDF's actuated set (yourdfpy resolves them automatically when
        # we update_cfg actuated values), so the role's full id list is
        # the same as the actuated id list here.
        self._joint_ids_by_role = self._actuated_ids_by_role

        # Per-actuated-joint limits, ordered by URDF actuated_joint_names.
        lows = np.zeros(self._n_actuated, dtype=np.float32)
        highs = np.zeros(self._n_actuated, dtype=np.float32)
        for i, name in enumerate(actuated_names):
            joint = urdf.joint_map[name]
            limit = getattr(joint, "limit", None)
            if limit is None:
                lows[i] = -np.inf
                highs[i] = np.inf
            else:
                lows[i] = float(getattr(limit, "lower", 0.0) or 0.0)
                highs[i] = float(getattr(limit, "upper", 0.0) or 0.0)
        self._lows = lows
        self._highs = highs

        self.joint_buffer = np.zeros(self._n_actuated, dtype=np.float32)

        self._default_qpos = torch.zeros(1, self._n_actuated, device=self.device)
        self._default_qvel = torch.zeros(1, self._n_actuated, device=self.device)

    # ---- joint lookups -------------------------------------------------- #

    def joint_ids_of(self, role: str) -> torch.Tensor:
        if role not in self._joint_ids_by_role:
            raise KeyError(
                f"role {role!r} not in handle.joints "
                f"(available: {list(self._joint_ids_by_role)})"
            )
        return self._joint_ids_by_role[role]

    def actuated_joint_ids_of(self, role: str) -> torch.Tensor:
        if role not in self._actuated_ids_by_role:
            raise KeyError(role)
        return self._actuated_ids_by_role[role]

    def actuated_joint_limits_of(self, role: str) -> tuple[torch.Tensor, torch.Tensor]:
        ids = self.actuated_joint_ids_of(role).detach().cpu().numpy()
        return (
            torch.from_numpy(self._lows[ids].copy()),
            torch.from_numpy(self._highs[ids].copy()),
        )

    # ---- state readers -------------------------------------------------- #

    @property
    def joint_pos(self) -> torch.Tensor:
        return torch.from_numpy(self.joint_buffer.copy()).unsqueeze(0)

    @property
    def joint_vel(self) -> torch.Tensor:
        return torch.zeros(1, self._n_actuated, device=self.device)

    @property
    def joint_pos_default(self) -> torch.Tensor:
        return self._default_qpos.clone()

    @property
    def joint_vel_default(self) -> torch.Tensor:
        return self._default_qvel.clone()

    # ---- command writers ------------------------------------------------ #

    def set_joint_position_target(
        self, targets: torch.Tensor, joint_ids: torch.Tensor
    ) -> None:
        tgt = targets.detach().cpu().numpy().reshape(-1)
        cols = joint_ids.detach().cpu().numpy().tolist()
        for i, col in enumerate(cols):
            self.joint_buffer[col] = float(tgt[i])

    def write_joint_state(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        del env_ids, joint_vel  # B=1, no dynamics
        self.joint_buffer[:] = joint_pos[0].detach().cpu().numpy()

    def reset_to_default(self) -> None:
        self.joint_buffer[:] = 0.0

    # ---- not implemented ------------------------------------------------ #

    def body_id_of(self, frame: str) -> int:
        raise NotImplementedError(_REPLAY_ONLY_MSG)

    def jacobi_body_id_of(self, frame: str) -> int:
        raise NotImplementedError(_REPLAY_ONLY_MSG)

    def ee_pose_b(self, frame: str | None = None) -> torch.Tensor:
        raise NotImplementedError(_REPLAY_ONLY_MSG)

    def ee_vel_b(self, frame: str | None = None) -> torch.Tensor:
        raise NotImplementedError(_REPLAY_ONLY_MSG)

    def mass_matrix(self, role: str) -> torch.Tensor:
        raise NotImplementedError(_REPLAY_ONLY_MSG)

    def jacobian(self, role: str, frame: str | None = None) -> torch.Tensor:
        raise NotImplementedError(_REPLAY_ONLY_MSG)

    def gravity(self, role: str) -> torch.Tensor:
        raise NotImplementedError(_REPLAY_ONLY_MSG)

    def set_joint_effort(self, efforts: torch.Tensor, joint_ids: torch.Tensor) -> None:
        raise NotImplementedError(_REPLAY_ONLY_MSG)

    def write_root_state(
        self,
        root_pose: torch.Tensor,
        root_velocity: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        raise NotImplementedError(_REPLAY_ONLY_MSG)

    def write_gains(
        self,
        role: str,
        stiffness: float | torch.Tensor,
        damping: float | torch.Tensor,
    ) -> None:
        raise NotImplementedError(_REPLAY_ONLY_MSG)
