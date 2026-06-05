"""Joint-space position-PD controller.

Trivial controller: `apply` writes
`default_pos + scale * command` as a position target for `role`'s
actuated joints. Covers the legacy `MinimalAR5RLEnv` use case and is
the default for the hand role in the OSC stack.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from linker_sim.backends.base import Robot


@dataclass
class JointPDControllerCfg:
    role: str = "hand"
    action_scale: float = 0.2
    # Gains default to the manifest's default_gains for the role.
    # Set explicit values here to override.
    stiffness: float | None = None
    damping: float | None = None


class JointPDController:
    """Position-target controller driving the role's actuated joints
    (mimic joints inherit via Isaac's implicit actuator group)."""

    command_dim: int  # set in attach based on joint count

    def __init__(self, cfg: JointPDControllerCfg | None = None):
        self.cfg = cfg or JointPDControllerCfg()
        self.role = self.cfg.role
        self._target: torch.Tensor | None = None
        self._joint_ids: torch.Tensor | None = None
        self.command_dim = 0

    def attach(self, robot: Robot) -> None:
        self._joint_ids = robot.actuated_joint_ids_of(self.role)  # type: ignore[attr-defined]
        self.command_dim = int(self._joint_ids.shape[0])
        if self.cfg.stiffness is not None and self.cfg.damping is not None:
            robot.write_gains(self.role, self.cfg.stiffness, self.cfg.damping)
        self._target = torch.zeros((robot.num_envs, self.command_dim), device=robot.device)

    def set_command(self, command: torch.Tensor, robot: Robot) -> None:
        self._target = command.clone()

    def apply(self, robot: Robot) -> None:
        assert self._joint_ids is not None and self._target is not None
        default = robot.joint_pos_default[:, self._joint_ids]
        robot.set_joint_position_target(default + self.cfg.action_scale * self._target, joint_ids=self._joint_ids)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if self._target is None:
            return
        if env_ids is None:
            self._target.zero_()
        else:
            self._target[env_ids] = 0.0
