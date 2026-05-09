"""Tests for `sim.tasks.reach.ReachTask`.

No Isaac — uses the FakeRobot / FakeBackend shared with test_base_env
to drive the task through reset/observe/reward/done.
"""

from __future__ import annotations

import torch

from sim.tasks.reach import ReachTask, ReachTaskCfg

import tests.test_base_env as _te
FakeBackend = _te.FakeBackend


def test_reach_task_observation_dim_matches_tensor():
    backend = FakeBackend()
    task = ReachTask(backend, ReachTaskCfg(action_dim=12))
    last_action = torch.zeros(backend.num_envs, task.action_dim)

    obs = task.observe(backend, last_action)
    assert obs.shape == (backend.num_envs, task.observation_dim)


def test_reach_task_reset_samples_target_inside_workspace():
    backend = FakeBackend()
    cfg = ReachTaskCfg(
        workspace_lo=(0.1, -0.1, 0.1),
        workspace_hi=(0.2, 0.1, 0.2),
    )
    task = ReachTask(backend, cfg)
    ids = torch.arange(backend.num_envs, dtype=torch.long)
    task.reset(backend, ids)

    lo = torch.tensor(cfg.workspace_lo)
    hi = torch.tensor(cfg.workspace_hi)
    t = task._target_pos
    assert (t >= lo - 1e-6).all() and (t <= hi + 1e-6).all()


def test_reach_task_reward_shaped_as_neg_distance():
    backend = FakeBackend()
    task = ReachTask(backend, ReachTaskCfg(
        pos_weight=1.0, ori_weight=0.0,
        action_penalty=0.0, joint_vel_penalty=0.0,
        action_dim=13,
    ))
    task._target_pos = torch.ones(backend.num_envs, 3)  # ee is at origin -> dist = sqrt(3)
    last_action = torch.zeros(backend.num_envs, task.action_dim)

    reward = task.reward(backend, last_action)
    expected = -torch.linalg.norm(torch.ones(3))
    assert torch.allclose(reward, expected.expand(backend.num_envs), atol=1e-5)


def test_reach_task_done_truncates_at_max_steps():
    backend = FakeBackend()
    task = ReachTask(backend, ReachTaskCfg())
    step_count = torch.tensor([99, 100, 101])
    terminated, truncated = task.done(backend, step_count, max_steps=101)
    # max_steps-1 = 100 -> envs 100 and 101 truncated
    assert truncated.tolist() == [False, True, True]
    assert not terminated.any()
