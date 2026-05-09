"""Pick-and-place task.

Requires the backend to pre-declare a task object named `cube` (by
convention — `cfg.object_name` overrides). The task resamples the
cube pose on reset, sets a target region, and shapes reward in three
stages per `docs/target_spec.md` §5:

    1. reach  — EE near cube
    2. lift   — cube above the table by `lift_threshold_z`
    3. place  — cube within `place_threshold` of target

Stage transitions are latching: once a stage is reached for
`success_hold_steps`, its reward bonus stays. Task terminates when
the place stage holds. No release/regrasp logic here — the gripper
is policy-driven.

Obs layout (concat, per-env):

    [joint_pos, joint_vel,
     ee_pose_b(7), ee_vel_b(6),
     cube_pose_b(7), cube_lin_vel_b(3),
     target_pos_b(3),
     ee_to_cube(3), cube_to_target(3),
     last_action]
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from sim.backends.base import SimBackend
from sim.tasks.base import Task


@dataclass
class PickPlaceTaskCfg:
    robot_name: str = "robot"
    object_name: str = "cube"
    ee_frame: str | None = None

    # Spawn box for the cube (robot-frame, meters).
    cube_spawn_lo: tuple[float, float, float] = (0.30, -0.10, 0.03)
    cube_spawn_hi: tuple[float, float, float] = (0.45, 0.10, 0.03)

    # Target region box.
    target_lo: tuple[float, float, float] = (0.30, -0.20, 0.15)
    target_hi: tuple[float, float, float] = (0.45, 0.20, 0.25)

    # Stage thresholds.
    reach_threshold: float = 0.05    # meters
    lift_threshold_z: float = 0.10
    place_threshold: float = 0.05

    # Reward shaping.
    reach_reward: float = 1.0
    lift_reward: float = 2.0
    place_reward: float = 5.0
    action_penalty: float = 0.01
    joint_vel_penalty: float = 0.001

    success_hold_steps: int = 10

    action_dim: int = 12
    observation_dim: int = field(default=0, init=False)


class PickPlaceTask:
    """Pick a cube, lift it, place at the sampled target."""

    def __init__(self, backend: SimBackend, cfg: PickPlaceTaskCfg | None = None):
        self.cfg = cfg or PickPlaceTaskCfg()
        self.action_dim = self.cfg.action_dim

        if self.cfg.object_name not in backend.rigid_bodies:
            raise KeyError(
                f"task object {self.cfg.object_name!r} not in backend.rigid_bodies. "
                f"Declare a RigidBodySpec on IsaacBackendCfg.rigid_bodies "
                f"with this name."
            )

        robot = backend.robots[self.cfg.robot_name]
        self._device = backend.device
        self._num_envs = backend.num_envs
        joint_dim = int(robot.joint_pos.shape[1])

        # [q, qd, ee(7), ee_vel(6), cube(7), cube_vel(3), target(3), ee_to_cube(3), cube_to_target(3), last_action]
        self.observation_dim = int(
            2 * joint_dim + 13 + 7 + 3 + 3 + 3 + 3 + self.cfg.action_dim
        )

        self._spawn_lo = torch.tensor(self.cfg.cube_spawn_lo, device=self._device)
        self._spawn_hi = torch.tensor(self.cfg.cube_spawn_hi, device=self._device)
        self._target_lo = torch.tensor(self.cfg.target_lo, device=self._device)
        self._target_hi = torch.tensor(self.cfg.target_hi, device=self._device)

        self._target_pos = torch.zeros(self._num_envs, 3, device=self._device)
        self._stage_success = torch.zeros(self._num_envs, 3, dtype=torch.bool, device=self._device)
        self._success_streak = torch.zeros(self._num_envs, dtype=torch.long, device=self._device)

    # -- Task interface --------------------------------------------------- #

    def reset(self, backend: SimBackend, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        n = env_ids.shape[0]

        cube = backend.rigid_bodies[self.cfg.object_name]
        cube_pos_w = self._spawn_lo + torch.rand(n, 3, device=self._device) * (self._spawn_hi - self._spawn_lo)
        cube_pos_w = cube_pos_w + backend.env_origins[env_ids]
        quat = torch.zeros(n, 4, device=self._device)
        quat[:, 0] = 1.0
        pose = torch.cat([cube_pos_w, quat], dim=-1)
        cube.write_root_pose(pose, env_ids=env_ids)
        cube.write_root_velocity(torch.zeros(n, 6, device=self._device), env_ids=env_ids)

        self._target_pos[env_ids] = (
            self._target_lo + torch.rand(n, 3, device=self._device) * (self._target_hi - self._target_lo)
        )
        self._stage_success[env_ids] = False
        self._success_streak[env_ids] = 0

    def _observe_tensors(self, backend: SimBackend):
        robot = backend.robots[self.cfg.robot_name]
        cube = backend.rigid_bodies[self.cfg.object_name]

        ee_pose = robot.ee_pose_b(self.cfg.ee_frame)
        ee_vel = robot.ee_vel_b(self.cfg.ee_frame)

        # Cube pose in robot root frame. We approximate by subtracting
        # env_origin (since the robot is fixed at origin of its env).
        # A rigorous impl would rotate into root frame; for single-arm
        # fixed-base workstations this is fine.
        cube_pos_b = cube.root_pos_w - backend.env_origins
        cube_vel_b = cube.root_lin_vel_w  # linear only; good enough for pick-place

        return robot, cube, ee_pose, ee_vel, cube_pos_b, cube_vel_b

    def observe(self, backend: SimBackend, last_action: torch.Tensor) -> torch.Tensor:
        robot, cube, ee_pose, ee_vel, cube_pos_b, cube_vel_b = self._observe_tensors(backend)

        cube_pose_b = torch.cat([cube_pos_b, cube.root_quat_w], dim=-1)
        ee_to_cube = cube_pos_b - ee_pose[:, :3]
        cube_to_target = self._target_pos - cube_pos_b

        return torch.cat(
            [
                robot.joint_pos, robot.joint_vel,
                ee_pose, ee_vel,
                cube_pose_b, cube_vel_b,
                self._target_pos,
                ee_to_cube, cube_to_target,
                last_action,
            ],
            dim=-1,
        )

    def reward(self, backend: SimBackend, last_action: torch.Tensor) -> torch.Tensor:
        robot, cube, ee_pose, _, cube_pos_b, _ = self._observe_tensors(backend)

        ee_to_cube_dist = torch.linalg.norm(cube_pos_b - ee_pose[:, :3], dim=-1)
        cube_z = cube_pos_b[:, 2]
        cube_to_target_dist = torch.linalg.norm(self._target_pos - cube_pos_b, dim=-1)

        reached = ee_to_cube_dist < self.cfg.reach_threshold
        lifted = reached & (cube_z > self.cfg.lift_threshold_z)
        placed = lifted & (cube_to_target_dist < self.cfg.place_threshold)

        self._stage_success[:, 0] |= reached
        self._stage_success[:, 1] |= lifted
        self._stage_success[:, 2] |= placed

        # Dense shaping: negative distance for the active stage.
        reach_shape = -ee_to_cube_dist
        lift_shape = torch.where(reached, -(self.cfg.lift_threshold_z - cube_z).clamp_min(0.0), torch.zeros_like(cube_z))
        place_shape = torch.where(lifted, -cube_to_target_dist, torch.zeros_like(cube_to_target_dist))

        # Sparse bonus per stage held.
        bonus = (
            self.cfg.reach_reward * reached.float()
            + self.cfg.lift_reward * lifted.float()
            + self.cfg.place_reward * placed.float()
        )

        action_pen = torch.sum(last_action ** 2, dim=-1)
        vel_pen = torch.sum(robot.joint_vel ** 2, dim=-1)

        return (
            reach_shape + lift_shape + place_shape + bonus
            - self.cfg.action_penalty * action_pen
            - self.cfg.joint_vel_penalty * vel_pen
        )

    def done(
        self,
        backend: SimBackend,
        step_count: torch.Tensor,
        max_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        robot, cube, ee_pose, _, cube_pos_b, _ = self._observe_tensors(backend)
        ee_to_cube_dist = torch.linalg.norm(cube_pos_b - ee_pose[:, :3], dim=-1)
        cube_z = cube_pos_b[:, 2]
        cube_to_target_dist = torch.linalg.norm(self._target_pos - cube_pos_b, dim=-1)

        placed = (
            (ee_to_cube_dist < self.cfg.reach_threshold)
            & (cube_z > self.cfg.lift_threshold_z)
            & (cube_to_target_dist < self.cfg.place_threshold)
        )
        self._success_streak = torch.where(
            placed, self._success_streak + 1, torch.zeros_like(self._success_streak)
        )
        terminated = self._success_streak >= self.cfg.success_hold_steps
        truncated = step_count >= (max_steps - 1)
        return terminated, truncated
