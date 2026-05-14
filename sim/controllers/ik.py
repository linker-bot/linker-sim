"""Damped-least-squares inverse kinematics controller.

Two action modes:

- ``abs_pose`` (default): command is a 7-dim absolute EE pose
  ``(pos(3), quat(4))`` in the robot root frame. The controller reads
  the current EE pose and computes a 6-dim pose error that it feeds
  into DLS. Matches DemoGrasp's ``armController=pose`` for RL pose-
  tracking.
- ``world_dpose``: command is a 6-dim delta ``(dpos, axis-angle)`` in
  the root frame, scaled per axis. Suitable for residual / delta-action
  policies.

Both modes use ``dq = J^T (J J^T + lambda^2 I)^{-1} dx`` (classical
DLS) so the mapping stays stable near singularities. Output is a joint
position target — PhysX's implicit PD does the tracking, so the arm's
``joint`` gain profile (or an optional per-controller override) sets
the effective stiffness.

Isaac-specific only because it needs ``robot.jacobian`` /
``robot.ee_pose_b``; a MuJoCo-side version with the same math lands
when that backend exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

import isaaclab.utils.math as math_utils

from sim.backends.base import Robot


@dataclass
class IkControllerCfg:
    role: str = "arm"
    frame: str | None = None

    mode: Literal["abs_pose", "world_dpose"] = "abs_pose"

    # Used only in world_dpose mode. abs_pose commands are consumed
    # unscaled — the task/policy layer is responsible for producing
    # in-workspace targets.
    action_scale_pos: float = 0.05
    action_scale_rot: float = 0.25

    damping: float = 0.05  # DLS lambda

    # Optional actuator PD override applied at attach(). None -> use
    # whatever gains the manifest's `joint` profile baked in.
    actuator_stiffness: float | None = None
    actuator_damping: float | None = None

    # When True, solve for (dx, dw) in the ROOT frame; else the current
    # EE frame. Root-frame deltas are what OSC uses, so we default here.
    command_in_root_frame: bool = True


class IkController:
    """DLS IK over the arm role. Emits joint position targets."""

    role: str
    command_dim: int

    def __init__(self, cfg: IkControllerCfg | None = None):
        self.cfg = cfg or IkControllerCfg()
        self.role = self.cfg.role
        self._joint_ids: torch.Tensor | None = None
        self._last_command: torch.Tensor | None = None
        self.command_dim = 0

    def attach(self, robot: Robot) -> None:
        self._joint_ids = robot.actuated_joint_ids_of(self.role)  # type: ignore[attr-defined]
        self.command_dim = 7 if self.cfg.mode == "abs_pose" else 6
        self._last_command = torch.zeros(
            (robot.num_envs, self.command_dim), device=robot.device
        )
        if self.cfg.actuator_stiffness is not None and self.cfg.actuator_damping is not None:
            robot.write_gains(  # type: ignore[attr-defined]
                self.role,
                self.cfg.actuator_stiffness,
                self.cfg.actuator_damping,
            )

    def set_command(self, command: torch.Tensor, robot: Robot) -> None:
        if self.cfg.mode == "abs_pose":
            cmd = command.clone()
            quat = cmd[:, 3:7]
            cmd[:, 3:7] = quat / (quat.norm(dim=-1, keepdim=True) + 1e-8)
            self._last_command = cmd
        else:
            scaled = command.clone()
            scaled[:, 0:3] *= self.cfg.action_scale_pos
            scaled[:, 3:6] *= self.cfg.action_scale_rot
            self._last_command = scaled

    def apply(self, robot: Robot) -> None:
        assert self._joint_ids is not None and self._last_command is not None

        J = robot.jacobian(self.role, self.cfg.frame)         # (B, 6, n)

        if self.cfg.mode == "abs_pose":
            ee = robot.ee_pose_b(self.cfg.frame)              # (B, 7)
            target_pos = self._last_command[:, 0:3]
            target_quat = self._last_command[:, 3:7]
            pos_err = target_pos - ee[:, 0:3]
            q_err = math_utils.quat_mul(target_quat, math_utils.quat_conjugate(ee[:, 3:7]))
            orn_err = math_utils.axis_angle_from_quat(q_err)
            dx = torch.cat([pos_err, orn_err], dim=-1)        # (B, 6)
        else:
            dx = self._last_command                            # (B, 6)

        # DLS: dq = J^T (J J^T + λ² I)^{-1} dx
        lam = self.cfg.damping
        JJt = J @ J.transpose(1, 2)                            # (B, 6, 6)
        eye = torch.eye(6, device=J.device).unsqueeze(0).expand_as(JJt)
        inv = torch.linalg.solve(JJt + (lam ** 2) * eye, dx.unsqueeze(-1))
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
