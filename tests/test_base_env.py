"""Tests for `sim.envs.base.BaseEnv` dispatch logic.

Uses a fake backend + fake robot + fake controllers so the loop,
decimation, action slicing, and reset flow are exercised without
bringing in Isaac. The fakes live in this file (they're small).
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from sim.envs.base import BaseEnv, BaseEnvCfg, Task
from sim.registry import ComponentRef, Gains, WorkstationHandle


def _make_handle(n_arm=7, n_hand=6) -> WorkstationHandle:
    return WorkstationHandle(
        name="fake",
        dir=None,  # type: ignore[arg-type]
        urdf_path=None,  # type: ignore[arg-type]
        mjcf_path=None,
        manifest_path=None,  # type: ignore[arg-type]
        recipe_path=None,  # type: ignore[arg-type]
        joints={"arm": [f"arm_j{i}" for i in range(n_arm)], "hand": [f"hand_j{i}" for i in range(n_hand)]},
        mimic_joints={"arm": [], "hand": []},
        frames={},
        ee_link="arm_tcp",
        ee_links={"arm": "arm_tcp"},
        base_link="arm_base",
        default_gains={"arm": Gains(1.0, 0.1), "hand": Gains(2.0, 0.2)},
        gain_profiles={},
        components={"arm": ComponentRef("arms/fake", "left", "x")},
        xrdf_paths={},
        raw_manifest={},
    )


class FakeRobot:
    def __init__(self, num_envs=3, device="cpu"):
        self.handle = _make_handle()
        self.num_envs = num_envs
        self.device = torch.device(device)
        n = sum(len(js) for js in self.handle.joints.values())
        self._n = n
        self._jp = torch.zeros(num_envs, n)
        self._jv = torch.zeros(num_envs, n)
        self._arm_ids = torch.arange(0, 7, dtype=torch.long)
        self._hand_ids = torch.arange(7, 13, dtype=torch.long)
        self.effort_log: list[tuple] = []
        self.pos_target_log: list[tuple] = []

    # identity helpers used by controllers
    def joint_ids_of(self, role):
        return {"arm": self._arm_ids, "hand": self._hand_ids}[role]

    def actuated_joint_ids_of(self, role):
        return self.joint_ids_of(role)

    def body_id_of(self, frame): return 0
    def jacobi_body_id_of(self, frame): return 0

    @property
    def joint_pos(self): return self._jp
    @property
    def joint_vel(self): return self._jv
    @property
    def joint_pos_default(self): return self._jp
    @property
    def joint_vel_default(self): return self._jv

    def ee_pose_b(self, frame=None):
        return torch.zeros(self.num_envs, 7)

    def ee_vel_b(self, frame=None):
        return torch.zeros(self.num_envs, 6)

    def mass_matrix(self, role):
        n = len(self.joint_ids_of(role))
        return torch.eye(n).expand(self.num_envs, n, n)

    def jacobian(self, role, frame=None):
        n = len(self.joint_ids_of(role))
        return torch.zeros(self.num_envs, 6, n)

    def gravity(self, role):
        return torch.zeros(self.num_envs, len(self.joint_ids_of(role)))

    def set_joint_effort(self, efforts, joint_ids):
        self.effort_log.append((efforts.clone(), joint_ids.clone()))

    def set_joint_position_target(self, targets, joint_ids):
        self.pos_target_log.append((targets.clone(), joint_ids.clone()))

    def write_joint_state(self, pos, vel, env_ids=None): pass
    def write_root_state(self, pose, vel, env_ids=None): pass
    def write_gains(self, role, stiffness, damping): pass


class FakeBackend:
    def __init__(self):
        self.num_envs = 3
        self.device = torch.device("cpu")
        self.dt = 1 / 60
        self.robots = {"robot": FakeRobot(num_envs=3)}
        self.rigid_bodies: dict = {}
        self.env_origins = torch.zeros(3, 3)
        self.step_count = 0
        self.reset_calls: list = []

    def step(self): self.step_count += 1
    def write_data(self): pass
    def reset(self, env_ids=None): self.reset_calls.append(env_ids)
    def close(self): pass


@dataclass
class _CountingController:
    role: str
    command_dim: int
    set_cmd_calls: int = 0
    apply_calls: int = 0
    reset_calls: int = 0

    def attach(self, robot): pass
    def set_command(self, command, robot): self.set_cmd_calls += 1
    def apply(self, robot): self.apply_calls += 1
    def reset(self, env_ids=None): self.reset_calls += 1


class _ZeroTask:
    observation_dim = 13
    action_dim = 10  # 6 + 4 (arbitrary)

    def reset(self, backend, env_ids): pass
    def observe(self, backend, last_action):
        return torch.zeros(backend.num_envs, self.observation_dim)
    def reward(self, backend, last_action):
        return torch.zeros(backend.num_envs)
    def done(self, backend, step_count, max_steps):
        terminated = torch.zeros_like(step_count, dtype=torch.bool)
        truncated = step_count >= max_steps - 1
        return terminated, truncated


def test_base_env_step_dispatches_action_slices_and_decimates():
    backend = FakeBackend()
    c0 = _CountingController(role="arm", command_dim=6)
    c1 = _CountingController(role="hand", command_dim=4)
    task = _ZeroTask()
    env = BaseEnv(backend, [c0, c1], task, cfg=BaseEnvCfg(decimation=4, episode_length_s=1.0))

    assert env.action_dim == 10
    assert env.observation_dim == 13

    obs, info = env.reset()
    assert obs.shape == (3, 13)

    action = torch.zeros(3, 10)
    obs, reward, terminated, truncated, info = env.step(action)

    # Each controller: set_command once, apply `decimation` times.
    assert c0.set_cmd_calls == 1 and c0.apply_calls == 4
    assert c1.set_cmd_calls == 1 and c1.apply_calls == 4
    assert backend.step_count == 4


def test_base_env_action_dim_mismatch_errors():
    backend = FakeBackend()
    c0 = _CountingController(role="arm", command_dim=6)

    class MismatchTask(_ZeroTask):
        action_dim = 99

    with pytest.raises(ValueError, match="action_dim"):
        BaseEnv(backend, [c0], MismatchTask(), cfg=BaseEnvCfg())


def test_base_env_resets_done_envs():
    backend = FakeBackend()
    c = _CountingController(role="arm", command_dim=10)
    task = _ZeroTask()

    class EpisodeLen1Task(_ZeroTask):
        def done(self, backend, step_count, max_steps):
            terminated = torch.zeros_like(step_count, dtype=torch.bool)
            # Truncate everyone after the first step.
            truncated = torch.ones_like(step_count, dtype=torch.bool)
            return terminated, truncated

    env = BaseEnv(backend, [c], EpisodeLen1Task(), cfg=BaseEnvCfg(decimation=1, episode_length_s=10.0))
    env.reset()
    action = torch.zeros(3, 10)
    env.step(action)
    # The env should have called backend.reset once during construction reset,
    # and once inside step (for the done envs).
    assert len(backend.reset_calls) >= 2
    assert c.reset_calls >= 2  # reset at env.reset + once for the done envs
