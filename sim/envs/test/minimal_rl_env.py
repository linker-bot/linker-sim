from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from sim.assets import make_ar5_l6_robot_cfg
from sim.envs.test.scene_cfg import TestDualSceneCfg, TestSceneCfg


@configclass
class MinimalAR5RLEnvCfg(DirectRLEnvCfg):
    """Minimal RL config on top of the current AR5/L6 scene."""

    # Core direct-RL settings
    decimation: int = 4
    episode_length_s: float = 8.0
    action_space: int = 1  # overwritten dynamically based on loaded robot(s)
    observation_space: int = 1  # overwritten dynamically based on loaded robot(s)

    # Scene
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, render_interval=4)
    scene: InteractiveSceneCfg = TestSceneCfg(num_envs=64, env_spacing=2.5)
    robot_side: Literal["left", "right", "both"] = "left"

    # Simple control/reward knobs
    action_scale: float = 0.25
    reset_joint_noise_scale: float = 0.02
    action_penalty_scale: float = 0.01
    joint_vel_penalty_scale: float = 0.001

    # Optional additive feed-forward torque offset on arm joints (first 7 DOFs per robot):
    # tau_cmd = tau_pd(implicit drive) + tau_ff_offset
    use_arm_feedforward_offset: bool = False
    arm_feedforward_offset: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


class MinimalAR5RLEnv(DirectRLEnv):
    """Minimal RL-ready environment using the existing table + AR5/L6 scene."""

    cfg: MinimalAR5RLEnvCfg

    def __init__(self, cfg: MinimalAR5RLEnvCfg, render_mode: str | None = None, **kwargs):
        # Resolve scene layout before parent constructor creates the scene.
        side = cfg.robot_side.lower()
        if side == "both":
            cfg.scene = TestDualSceneCfg(num_envs=cfg.scene.num_envs, env_spacing=cfg.scene.env_spacing)
            self._robot_names = ["robot_left", "robot_right"]
        else:
            cfg.scene = TestSceneCfg(num_envs=cfg.scene.num_envs, env_spacing=cfg.scene.env_spacing)
            cfg.scene.robot = make_ar5_l6_robot_cfg(side=side, prim_path="{ENV_REGEX_NS}/Robot")
            self._robot_names = ["robot"]
        self._robot_side = side
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

    def _setup_scene(self):
        # Scene entities are already constructed via cfg.scene.
        pass

    def _configure_gym_env_spaces(self):
        # Scene is available here; infer action/obs dims from loaded articulation(s).
        self._robots = [self.scene[name] for name in self._robot_names]
        self._joint_dims = [robot.data.default_joint_pos.shape[1] for robot in self._robots]
        self._action_dim = int(sum(self._joint_dims))
        self._obs_dim = int(2 * self._action_dim)  # [joint_pos, joint_vel]

        self.cfg.action_space = self._action_dim
        self.cfg.observation_space = self._obs_dim
        super()._configure_gym_env_spaces()

        self._actions = torch.zeros((self.num_envs, self._action_dim), device=self.device)

        # Pre-build per-robot feed-forward torque tensors (all envs) for fast per-step apply.
        arm_ff = torch.tensor(self.cfg.arm_feedforward_offset, device=self.device, dtype=torch.float32)
        if arm_ff.numel() != 7:
            raise ValueError(
                f"arm_feedforward_offset must have 7 values, got {arm_ff.numel()}: {self.cfg.arm_feedforward_offset}"
            )
        self._arm_ff_efforts: list[torch.Tensor] = []
        for joint_dim in self._joint_dims:
            ff = torch.zeros((self.num_envs, joint_dim), device=self.device, dtype=torch.float32)
            ff[:, : min(7, joint_dim)] = arm_ff[: min(7, joint_dim)]
            self._arm_ff_efforts.append(ff)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = torch.clamp(actions, -1.0, 1.0)

    def _apply_action(self):
        cursor = 0
        for robot, joint_dim, arm_ff in zip(
            self._robots, self._joint_dims, self._arm_ff_efforts, strict=True
        ):
            act = self._actions[:, cursor : cursor + joint_dim]
            cursor += joint_dim

            target = robot.data.default_joint_pos + self.cfg.action_scale * act
            robot.set_joint_position_target(target)
            # Add constant feed-forward torque offset on top of implicit PD drive.
            if self.cfg.use_arm_feedforward_offset:
                robot.set_joint_effort_target(arm_ff)

    def _get_observations(self):
        obs_chunks = []
        for robot in self._robots:
            obs_chunks.append(robot.data.joint_pos)
            obs_chunks.append(robot.data.joint_vel)
        obs = torch.cat(obs_chunks, dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        action_penalty = torch.sum(self._actions**2, dim=-1)

        vel_penalty = 0.0
        for robot in self._robots:
            vel_penalty = vel_penalty + torch.sum(robot.data.joint_vel**2, dim=-1)

        reward = 1.0 - self.cfg.action_penalty_scale * action_penalty - self.cfg.joint_vel_penalty_scale * vel_penalty
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        time_out = self.episode_length_buf >= (self.max_episode_length - 1)
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

        super()._reset_idx(env_ids)

        for robot in self._robots:
            root_state = robot.data.default_root_state[env_ids_t].clone()
            root_state[:, :3] += self.scene.env_origins[env_ids_t]
            robot.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids_t)
            robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids_t)

            joint_pos = robot.data.default_joint_pos[env_ids_t].clone()
            joint_vel = robot.data.default_joint_vel[env_ids_t].clone()
            joint_pos += self.cfg.reset_joint_noise_scale * (torch.rand_like(joint_pos) - 0.5)
            robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids_t)

        self._actions[env_ids_t] = 0.0
