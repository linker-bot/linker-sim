"""Bimanual reach task: drive two EEs to independent sampled targets.

Parameterized over `arms`: a list of per-arm specs (role, ee_frame,
workspace). Each arm samples its own target on reset; reward is the
sum of per-arm shaped reach
rewards; termination requires every arm's success streak to hit
`success_hold_steps`.

Obs layout (concat, per-env):

    shared  : [joint_pos, joint_vel]
    per-arm : [ee_pose_b(7), ee_vel_b(6), target_pose_b(7),
               pos_err(3), quat_dot(1)]    # x N_arms
    shared  : [last_action]

Action layout matches the controller ordering in
`configs/controller/osc_bimanual.yaml`:
    [arm_left(6), hand_left(6), arm_right(6), hand_right(6)]
The task is agnostic to action internals — it reads `last_action` as
an opaque blob and only uses its magnitude for the action penalty.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from sim.backends.base import SimBackend
from sim.tasks.base import Task


def _rpy_to_quat_wxyz(rpy: torch.Tensor) -> torch.Tensor:
    """`(N, 3)` roll/pitch/yaw -> `(N, 4)` wxyz quaternion (XYZ Tait-Bryan)."""
    r, p, y = rpy.unbind(-1)
    cr, sr = torch.cos(r * 0.5), torch.sin(r * 0.5)
    cp, sp = torch.cos(p * 0.5), torch.sin(p * 0.5)
    cy, sy = torch.cos(y * 0.5), torch.sin(y * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y_ = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return torch.stack([w, x, y_, z], dim=-1)


@dataclass
class ArmSpec:
    """Per-arm reach target spec."""

    role: str = "arm_left"
    ee_frame: str = "arm_left:tool0"
    workspace_lo: tuple[float, float, float] = (0.20, -0.25, 0.15)
    workspace_hi: tuple[float, float, float] = (0.55, 0.25, 0.55)
    orientation_range: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class BimanualReachTaskCfg:
    robot_name: str = "robot"
    arms: list[ArmSpec] = field(
        default_factory=lambda: [
            ArmSpec(role="arm_left", ee_frame="arm_left:tool0"),
            ArmSpec(
                role="arm_right",
                ee_frame="arm_right:tool0",
                workspace_lo=(0.20, -0.25, 0.15),
                workspace_hi=(0.55, 0.25, 0.55),
            ),
        ]
    )

    # Reward shaping (shared across arms; summed).
    pos_weight: float = 2.0
    ori_weight: float = 0.5
    action_penalty: float = 0.01
    joint_vel_penalty: float = 0.001

    # Per-arm success thresholds (same threshold for both arms).
    success_pos_threshold: float = 0.02
    success_ori_threshold_rad: float = 0.1745
    success_hold_steps: int = 5

    # Sum of controller command_dims. Cross-checked by BaseEnv.
    action_dim: int = 24

    observation_dim: int = field(default=0, init=False)


class BimanualReachTask:
    """Two-arm reach: each arm hits its own target."""

    def __init__(self, backend: SimBackend, cfg: BimanualReachTaskCfg | None = None):
        self.cfg = cfg or BimanualReachTaskCfg()
        if len(self.cfg.arms) < 2:
            raise ValueError(
                f"BimanualReachTask expects >= 2 arms in cfg.arms, got "
                f"{len(self.cfg.arms)}."
            )

        self.action_dim = self.cfg.action_dim

        robot = backend.robots[self.cfg.robot_name]
        self._robot_name = self.cfg.robot_name
        self._device = backend.device
        self._num_envs = backend.num_envs
        self._joint_dim = int(robot.joint_pos.shape[1])
        self._n_arms = len(self.cfg.arms)

        # [q, qd] (shared) + N * [ee(7) + ee_vel(6) + target(7) + err(4)] + last_action
        self.observation_dim = int(
            2 * self._joint_dim
            + self._n_arms * (7 + 6 + 7 + 4)
            + self.cfg.action_dim
        )

        self._ws_lo = torch.stack(
            [torch.tensor(a.workspace_lo, device=self._device) for a in self.cfg.arms]
        )  # (N, 3)
        self._ws_hi = torch.stack(
            [torch.tensor(a.workspace_hi, device=self._device) for a in self.cfg.arms]
        )
        self._ori_range = torch.stack(
            [torch.tensor(a.orientation_range, device=self._device) for a in self.cfg.arms]
        )

        # Per-env, per-arm target pose + success streak.
        self._target_pos = torch.zeros(self._num_envs, self._n_arms, 3, device=self._device)
        self._target_quat = torch.zeros(self._num_envs, self._n_arms, 4, device=self._device)
        self._target_quat[..., 0] = 1.0
        self._success_streak = torch.zeros(
            self._num_envs, dtype=torch.long, device=self._device
        )

    # -- Task interface --------------------------------------------------- #

    def reset(self, backend: SimBackend, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        n = env_ids.shape[0]
        # (n, N_arms, 3)
        rand = torch.rand(n, self._n_arms, 3, device=self._device)
        self._target_pos[env_ids] = self._ws_lo + rand * (self._ws_hi - self._ws_lo)

        for i, arm in enumerate(self.cfg.arms):
            if self._ori_range[i].any():
                rpy = (
                    torch.rand(n, 3, device=self._device) - 0.5
                ) * 2.0 * self._ori_range[i]
                self._target_quat[env_ids, i] = _rpy_to_quat_wxyz(rpy)
            else:
                self._target_quat[env_ids, i] = torch.tensor(
                    [1.0, 0.0, 0.0, 0.0], device=self._device
                )

        self._success_streak[env_ids] = 0

    def _per_arm_ee(self, backend: SimBackend) -> tuple[torch.Tensor, torch.Tensor]:
        """Stacked (num_envs, N_arms, 7) ee_pose and (.., 6) ee_vel."""
        robot = backend.robots[self._robot_name]
        poses = torch.stack(
            [robot.ee_pose_b(arm.ee_frame) for arm in self.cfg.arms], dim=1
        )
        vels = torch.stack(
            [robot.ee_vel_b(arm.ee_frame) for arm in self.cfg.arms], dim=1
        )
        return poses, vels

    def observe(self, backend: SimBackend, last_action: torch.Tensor) -> torch.Tensor:
        robot = backend.robots[self._robot_name]
        poses, vels = self._per_arm_ee(backend)  # (B, N, 7), (B, N, 6)

        target_pose = torch.cat([self._target_pos, self._target_quat], dim=-1)  # (B, N, 7)
        pos_err = poses[..., :3] - self._target_pos  # (B, N, 3)
        quat_dot = torch.sum(
            poses[..., 3:7] * self._target_quat, dim=-1, keepdim=True
        )  # (B, N, 1)

        # Flatten per-arm features into (B, N*24).
        per_arm = torch.cat([poses, vels, target_pose, pos_err, quat_dot], dim=-1)
        per_arm_flat = per_arm.reshape(self._num_envs, -1)

        return torch.cat(
            [robot.joint_pos, robot.joint_vel, per_arm_flat, last_action],
            dim=-1,
        )

    def _pos_and_ori_err(
        self, backend: SimBackend
    ) -> tuple[torch.Tensor, torch.Tensor]:
        poses, _ = self._per_arm_ee(backend)  # (B, N, 7)
        pos_err = torch.linalg.norm(poses[..., :3] - self._target_pos, dim=-1)  # (B, N)
        quat_dot = (
            torch.sum(poses[..., 3:7] * self._target_quat, dim=-1).abs().clamp_(max=1.0)
        )
        ori_err = 2.0 * torch.acos(quat_dot)  # (B, N)
        return pos_err, ori_err

    def reward(self, backend: SimBackend, last_action: torch.Tensor) -> torch.Tensor:
        robot = backend.robots[self._robot_name]
        pos_err, ori_err = self._pos_and_ori_err(backend)

        action_pen = torch.sum(last_action ** 2, dim=-1)
        vel_pen = torch.sum(robot.joint_vel ** 2, dim=-1)

        # Sum shaped terms across arms; the two penalty terms are
        # already shared-state scalars and are added once.
        return (
            -self.cfg.pos_weight * pos_err.sum(dim=-1)
            - self.cfg.ori_weight * ori_err.sum(dim=-1)
            - self.cfg.action_penalty * action_pen
            - self.cfg.joint_vel_penalty * vel_pen
        )

    def done(
        self,
        backend: SimBackend,
        step_count: torch.Tensor,
        max_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pos_err, ori_err = self._pos_and_ori_err(backend)
        in_threshold = (pos_err < self.cfg.success_pos_threshold) & (
            ori_err < self.cfg.success_ori_threshold_rad
        )
        # Only count the streak when BOTH arms are in-threshold.
        all_in = in_threshold.all(dim=-1)
        self._success_streak = torch.where(
            all_in, self._success_streak + 1, torch.zeros_like(self._success_streak)
        )
        terminated = self._success_streak >= self.cfg.success_hold_steps
        truncated = step_count >= (max_steps - 1)
        return terminated, truncated
