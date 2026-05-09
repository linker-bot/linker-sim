"""`BaseEnv` — gym-style env over `SimBackend` + `Controller`s + `Task`.

Owns the control loop, decimation, reset bookkeeping, and episode-length
tracking. Does NOT inherit from IsaacLab's `DirectRLEnv` or
`ManagerBasedRLEnv` (D9): the backend composes `SimulationContext` +
`Articulation` directly; this env just drives it.

Step loop::

    env.step(action)
      for each controller: set_command(action_slice, robot)
      for _ in range(decimation):
        for each controller: apply(robot)
        backend.write_data()
        backend.step()
      # post-physics: observe, reward, done
      # if any env done: reset(env_ids)

The `Task` Protocol (obs/reward/done) lives in `sim.tasks.base`;
concrete tasks live under `sim/tasks/`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sim.backends.base import Robot, SimBackend
from sim.controllers.base import Controller
from sim.tasks.base import Task


@dataclass
class BaseEnvCfg:
    robot_name: str = "robot"
    decimation: int = 4
    episode_length_s: float = 8.0
    reset_joint_noise_scale: float = 0.02


class BaseEnv:
    """Gym-style env over a `SimBackend`.

    Parameters
    ----------
    backend:
        A running `SimBackend` (Isaac today; MuJoCo when PR #1b lands).
    controllers:
        Ordered list of controllers driving the env's single robot.
        Actions are sliced in this order: `action[:, :c0.command_dim]`
        goes to `c0.set_command`, next `c1.command_dim` to `c1`, etc.
    task:
        Task providing obs / reward / done.
    cfg:
        Shared env-level knobs (decimation, episode length, reset noise).
    """

    def __init__(
        self,
        backend: SimBackend,
        controllers: list[Controller],
        task: Task,
        cfg: BaseEnvCfg | None = None,
    ):
        self.backend = backend
        self.controllers = controllers
        self.task = task
        self.cfg = cfg or BaseEnvCfg()

        if self.cfg.robot_name not in backend.robots:
            raise KeyError(
                f"robot {self.cfg.robot_name!r} not in backend.robots "
                f"(available: {list(backend.robots)})"
            )
        self._robot: Robot = backend.robots[self.cfg.robot_name]

        for c in controllers:
            c.attach(self._robot)
        self._action_slices = self._compute_action_slices(controllers)

        action_dim = int(sum(c.command_dim for c in controllers))
        if task.action_dim != action_dim:
            raise ValueError(
                f"Task declares action_dim={task.action_dim} but controllers "
                f"sum to {action_dim}. Align the two before running."
            )
        self.action_dim = action_dim
        self.observation_dim = task.observation_dim

        self._step_count = torch.zeros(backend.num_envs, dtype=torch.long, device=backend.device)
        self._max_steps = max(1, int(self.cfg.episode_length_s / (backend.dt * self.cfg.decimation)))
        self._last_action = torch.zeros((backend.num_envs, action_dim), device=backend.device)

    # -- gym API --------------------------------------------------------- #

    def reset(
        self,
        seed: int | None = None,
        env_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        if seed is not None:
            torch.manual_seed(seed)
        if env_ids is None:
            env_ids = torch.arange(
                self.backend.num_envs, dtype=torch.long, device=self.backend.device
            )
        env_ids = env_ids.to(device=self.backend.device, dtype=torch.long)

        self.backend.reset(env_ids=env_ids)
        self._jitter_joints(env_ids)

        for c in self.controllers:
            c.reset(env_ids)
        self.task.reset(self.backend, env_ids)

        self._step_count[env_ids] = 0
        self._last_action[env_ids] = 0.0

        obs = self.task.observe(self.backend, self._last_action)
        return obs, {}

    def step(
        self, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        action = torch.clamp(action, -1.0, 1.0)
        self._last_action = action

        for c, sl in zip(self.controllers, self._action_slices, strict=True):
            c.set_command(action[:, sl], self._robot)

        for _ in range(self.cfg.decimation):
            for c in self.controllers:
                c.apply(self._robot)
            self.backend.write_data()
            self.backend.step()

        self._step_count += 1
        obs = self.task.observe(self.backend, self._last_action)
        reward = self.task.reward(self.backend, self._last_action)
        terminated, truncated = self.task.done(self.backend, self._step_count, self._max_steps)

        done_mask = terminated | truncated
        if done_mask.any():
            done_env_ids = torch.nonzero(done_mask, as_tuple=False).squeeze(-1)
            self.backend.reset(env_ids=done_env_ids)
            self._jitter_joints(done_env_ids)
            for c in self.controllers:
                c.reset(done_env_ids)
            self.task.reset(self.backend, done_env_ids)
            self._step_count[done_env_ids] = 0
            self._last_action[done_env_ids] = 0.0

        return obs, reward, terminated, truncated, {}

    # -- helpers --------------------------------------------------------- #

    @staticmethod
    def _compute_action_slices(controllers: list[Controller]) -> list[slice]:
        slices: list[slice] = []
        cursor = 0
        for c in controllers:
            slices.append(slice(cursor, cursor + c.command_dim))
            cursor += c.command_dim
        return slices

    def _jitter_joints(self, env_ids: torch.Tensor) -> None:
        if self.cfg.reset_joint_noise_scale <= 0.0:
            return
        robot = self._robot
        jp = robot.joint_pos_default[env_ids].clone()
        jv = robot.joint_vel_default[env_ids].clone()
        jp += self.cfg.reset_joint_noise_scale * (torch.rand_like(jp) - 0.5)
        robot.write_joint_state(jp, jv, env_ids=env_ids)

    @property
    def robot(self) -> Robot:
        return self._robot

    @property
    def num_envs(self) -> int:
        return self.backend.num_envs

    @property
    def device(self) -> torch.device:
        return self.backend.device

    @property
    def max_steps(self) -> int:
        return self._max_steps
