"""Reach task: drive the EE to a sampled target pose.

Target is sampled in a workspace box in the robot's root frame. Reward
is a shaped combination of position error, orientation error, action
magnitude, and joint velocity. Success thresholds match
`docs/target_spec.md` §5 (<= 2 cm, <= 10°).

Obs layout (concat, per-env):

    [joint_pos, joint_vel,                # robot state
     ee_pose_b (7), ee_vel_b (6),         # EE state in root frame
     target_pose_b (7),                   # sampled target
     target_pose_err (3 pos + 1 quat-dot),# error terms
     last_action]
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from sim.backends.base import SimBackend
from sim.tasks.base import Task


@dataclass
class ReachTaskCfg:
    robot_name: str = "robot"
    ee_frame: str | None = None  # None -> robot.handle.ee_link

    # Workspace box (robot-frame, meters). Targets are uniform-sampled
    # here. Defaults are a loose box above the table, biased forward.
    workspace_lo: tuple[float, float, float] = (0.20, -0.25, 0.15)
    workspace_hi: tuple[float, float, float] = (0.55, 0.25, 0.55)

    # Orientation: sampled as a small random rotation around each axis
    # in +/- this range (radians). Default 0 -> target orientation ==
    # default EE orientation at reset.
    orientation_range: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Reward shaping.
    pos_weight: float = 2.0
    ori_weight: float = 0.5
    action_penalty: float = 0.01
    joint_vel_penalty: float = 0.001

    # Success/termination thresholds (from docs/target_spec.md).
    success_pos_threshold: float = 0.02   # 2 cm
    success_ori_threshold_rad: float = 0.1745  # ~10 deg
    # Minimum consecutive steps inside the threshold before we call it
    # terminated=True (stability check, per target_spec §5).
    success_hold_steps: int = 5

    # Width of the action vector the controllers consume. Cross-checked
    # against attached controllers by `BaseEnv`.
    action_dim: int = 12

    # Set by __post_init__ based on the attached robot.
    observation_dim: int = field(default=0, init=False)


class ReachTask:
    """Sim-agnostic reach task: drive EE to a sampled pose."""

    def __init__(self, backend: SimBackend, cfg: ReachTaskCfg | None = None):
        self.cfg = cfg or ReachTaskCfg()
        self.action_dim = self.cfg.action_dim

        robot = backend.robots[self.cfg.robot_name]
        self._robot_name = self.cfg.robot_name
        self._device = backend.device
        self._num_envs = backend.num_envs
        self._joint_dim = int(robot.joint_pos.shape[1])

        # [q, qd, ee_pose(7), ee_vel(6), target(7), err(4), last_action]
        self.observation_dim = int(
            2 * self._joint_dim + 13 + 7 + 4 + self.cfg.action_dim
        )

        self._ws_lo = torch.tensor(self.cfg.workspace_lo, device=self._device)
        self._ws_hi = torch.tensor(self.cfg.workspace_hi, device=self._device)
        self._ori_range = torch.tensor(self.cfg.orientation_range, device=self._device)

        # Per-env target pose + success streak counter.
        self._target_pos = torch.zeros(self._num_envs, 3, device=self._device)
        self._target_quat = torch.zeros(self._num_envs, 4, device=self._device)
        self._target_quat[:, 0] = 1.0  # identity wxyz
        self._success_streak = torch.zeros(self._num_envs, dtype=torch.long, device=self._device)

    # -- Task interface --------------------------------------------------- #

    def reset(self, backend: SimBackend, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        lo, hi = self._ws_lo, self._ws_hi
        rand = torch.rand(env_ids.shape[0], 3, device=self._device)
        self._target_pos[env_ids] = lo + rand * (hi - lo)

        if self._ori_range.any():
            rpy = (torch.rand(env_ids.shape[0], 3, device=self._device) - 0.5) * 2.0 * self._ori_range
            self._target_quat[env_ids] = _rpy_to_quat_wxyz(rpy)
        else:
            self._target_quat[env_ids] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device)

        self._success_streak[env_ids] = 0

    def observe(self, backend: SimBackend, last_action: torch.Tensor) -> torch.Tensor:
        robot = backend.robots[self._robot_name]
        ee_pose = robot.ee_pose_b(self.cfg.ee_frame)
        ee_vel = robot.ee_vel_b(self.cfg.ee_frame)

        target_pose = torch.cat([self._target_pos, self._target_quat], dim=-1)
        pos_err = ee_pose[:, :3] - self._target_pos
        # Quaternion dot product as a cheap orientation similarity
        # scalar in [-1, 1]; full error vector isn't needed for obs.
        quat_dot = torch.sum(ee_pose[:, 3:7] * self._target_quat, dim=-1, keepdim=True)

        return torch.cat(
            [robot.joint_pos, robot.joint_vel, ee_pose, ee_vel, target_pose, pos_err, quat_dot, last_action],
            dim=-1,
        )

    def reward(self, backend: SimBackend, last_action: torch.Tensor) -> torch.Tensor:
        robot = backend.robots[self._robot_name]
        ee_pose = robot.ee_pose_b(self.cfg.ee_frame)

        pos_err = torch.linalg.norm(ee_pose[:, :3] - self._target_pos, dim=-1)
        # Orientation error: 2*acos(|q . q_target|). Clamp to avoid NaN.
        quat_dot = torch.sum(ee_pose[:, 3:7] * self._target_quat, dim=-1).abs().clamp_(max=1.0)
        ori_err = 2.0 * torch.acos(quat_dot)

        action_pen = torch.sum(last_action ** 2, dim=-1)
        vel_pen = torch.sum(robot.joint_vel ** 2, dim=-1)

        return (
            -self.cfg.pos_weight * pos_err
            - self.cfg.ori_weight * ori_err
            - self.cfg.action_penalty * action_pen
            - self.cfg.joint_vel_penalty * vel_pen
        )

    def done(
        self,
        backend: SimBackend,
        step_count: torch.Tensor,
        max_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        robot = backend.robots[self._robot_name]
        ee_pose = robot.ee_pose_b(self.cfg.ee_frame)

        pos_err = torch.linalg.norm(ee_pose[:, :3] - self._target_pos, dim=-1)
        quat_dot = torch.sum(ee_pose[:, 3:7] * self._target_quat, dim=-1).abs().clamp_(max=1.0)
        ori_err = 2.0 * torch.acos(quat_dot)

        in_threshold = (pos_err < self.cfg.success_pos_threshold) & (
            ori_err < self.cfg.success_ori_threshold_rad
        )
        self._success_streak = torch.where(
            in_threshold, self._success_streak + 1, torch.zeros_like(self._success_streak)
        )
        terminated = self._success_streak >= self.cfg.success_hold_steps
        truncated = step_count >= (max_steps - 1)
        return terminated, truncated


# -------------------- helpers ------------------------------------------- #


def _rpy_to_quat_wxyz(rpy: torch.Tensor) -> torch.Tensor:
    """`(N, 3)` roll/pitch/yaw -> `(N, 4)` wxyz quaternion.

    Follows the XYZ Tait-Bryan convention (same as IsaacLab's
    `math_utils.quat_from_euler_xyz`). Reimplemented here so the task
    stays backend-agnostic."""
    r, p, y = rpy.unbind(-1)
    cr, sr = torch.cos(r * 0.5), torch.sin(r * 0.5)
    cp, sp = torch.cos(p * 0.5), torch.sin(p * 0.5)
    cy, sy = torch.cos(y * 0.5), torch.sin(y * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y_ = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return torch.stack([w, x, y_, z], dim=-1)
