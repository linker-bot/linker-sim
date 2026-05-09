"""Damped-least-squares inverse kinematics controller.

Takes a `(B, 6)` delta-pose command (translation + axis-angle) in the
robot root frame and computes joint position targets that drive the
configured frame toward the commanded pose over the decimation window.

Uses `J^T (J J^T + lambda^2 I)^{-1}` (classical DLS) so the mapping
stays stable near singularities. Single-step — the env should be
running at a high enough physics rate that accumulated deltas track
the policy's intent.

Isaac-specific only because it needs `robot.jacobian` semantics; a
MuJoCo-side version with the same math lands when that backend exists.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sim.backends.base import Robot


@dataclass
class IkControllerCfg:
    role: str = "arm"
    frame: str | None = None

    action_scale_pos: float = 0.05
    action_scale_rot: float = 0.25

    damping: float = 0.05  # DLS lambda
    # When True, solve for (dx, dw) in the ROOT frame; else the current
    # EE frame. Root-frame deltas are what OSC uses, so we default here.
    command_in_root_frame: bool = True


class IkController:
    """DLS IK over the arm role. Emits joint position targets."""

    role: str
    command_dim: int = 6

    def __init__(self, cfg: IkControllerCfg | None = None):
        self.cfg = cfg or IkControllerCfg()
        self.role = self.cfg.role
        self._joint_ids: torch.Tensor | None = None
        self._last_command: torch.Tensor | None = None

    def attach(self, robot: Robot) -> None:
        self._joint_ids = robot.actuated_joint_ids_of(self.role)  # type: ignore[attr-defined]
        self._last_command = torch.zeros((robot.num_envs, 6), device=robot.device)

    def set_command(self, command: torch.Tensor, robot: Robot) -> None:
        scaled = command.clone()
        scaled[:, 0:3] *= self.cfg.action_scale_pos
        scaled[:, 3:6] *= self.cfg.action_scale_rot
        self._last_command = scaled

    def apply(self, robot: Robot) -> None:
        assert self._joint_ids is not None and self._last_command is not None

        J = robot.jacobian(self.role, self.cfg.frame)         # (B, 6, n)
        delta = self._last_command                             # (B, 6)

        # DLS: dq = J^T (J J^T + λ² I)^{-1} dx
        lam = self.cfg.damping
        JJt = J @ J.transpose(1, 2)                            # (B, 6, 6)
        eye = torch.eye(6, device=J.device).unsqueeze(0).expand_as(JJt)
        inv = torch.linalg.solve(JJt + (lam ** 2) * eye, delta.unsqueeze(-1))
        dq = (J.transpose(1, 2) @ inv).squeeze(-1)             # (B, n)

        current = robot.joint_pos[:, self._joint_ids]
        target = current + dq
        robot.set_joint_position_target(target, joint_ids=self._joint_ids)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if self._last_command is None:
            return
        if env_ids is None:
            self._last_command.zero_()
        else:
            self._last_command[env_ids] = 0.0
