"""Tests for `sim.tasks.bimanual_reach.BimanualReachTask`.

No Isaac: uses an inline fake backend with a bimanual handle and an
ee_pose_b that returns different values per arm frame. Focus is on
dim-matching + reward-summing semantics — the actual kinematics come
from the Isaac backend at runtime.
"""

from __future__ import annotations

import torch

from sim.registry import ComponentRef, Gains, WorkstationHandle
from sim.tasks.bimanual_reach import ArmSpec, BimanualReachTask, BimanualReachTaskCfg


def _bimanual_handle() -> WorkstationHandle:
    roles = {
        "base": [],
        "arm_left": [f"arm_left_j{i}" for i in range(7)],
        "arm_right": [f"arm_right_j{i}" for i in range(7)],
        "hand_left": [f"hand_left_j{i}" for i in range(6)],
        "hand_right": [f"hand_right_j{i}" for i in range(6)],
    }
    gains = {r: Gains(1.0, 0.1) for r in roles if roles[r]}
    return WorkstationHandle(
        name="fake_bimanual",
        dir=None,  # type: ignore[arg-type]
        urdf_path=None,  # type: ignore[arg-type]
        mjcf_path=None,
        manifest_path=None,  # type: ignore[arg-type]
        recipe_path=None,  # type: ignore[arg-type]
        joints=roles,
        mimic_joints={r: [] for r in roles},
        frames={"arm_left:tool0": "L_tcp", "arm_right:tool0": "R_tcp"},
        ee_link="L_tcp",
        ee_links={"arm_left": "L_tcp", "arm_right": "R_tcp"},
        base_link="base_link",
        default_gains=gains,
        gain_profiles={},
        components={"arm_left": ComponentRef("arms/ar5", "left", "x")},
        raw_manifest={},
    )


class _FakeBimanualRobot:
    """Reads per-arm EE pose/vel from a dict keyed by frame string."""

    def __init__(self, num_envs=3):
        self.handle = _bimanual_handle()
        self.num_envs = num_envs
        self.device = torch.device("cpu")
        total = sum(len(js) for js in self.handle.joints.values())
        self._jp = torch.zeros(num_envs, total)
        self._jv = torch.zeros(num_envs, total)

        # Default EE poses: left arm at (0.3, 0.1, 0.3), right at (0.3, -0.1, 0.3),
        # both with identity orientation.
        self._ee_poses: dict[str, torch.Tensor] = {
            "arm_left:tool0": torch.tensor([0.3, 0.1, 0.3, 1.0, 0.0, 0.0, 0.0]).expand(
                num_envs, 7
            ).clone(),
            "arm_right:tool0": torch.tensor([0.3, -0.1, 0.3, 1.0, 0.0, 0.0, 0.0]).expand(
                num_envs, 7
            ).clone(),
        }
        self._ee_vels: dict[str, torch.Tensor] = {
            k: torch.zeros(num_envs, 6) for k in self._ee_poses
        }

    @property
    def joint_pos(self):
        return self._jp

    @property
    def joint_vel(self):
        return self._jv

    def ee_pose_b(self, frame=None):
        return self._ee_poses[frame]

    def ee_vel_b(self, frame=None):
        return self._ee_vels[frame]


class _FakeBimanualBackend:
    def __init__(self):
        self.num_envs = 3
        self.device = torch.device("cpu")
        self.dt = 1 / 60
        self.robots = {"robot": _FakeBimanualRobot(num_envs=3)}
        self.rigid_bodies: dict = {}
        self.env_origins = torch.zeros(3, 3)


def test_bimanual_reach_dims():
    backend = _FakeBimanualBackend()
    task = BimanualReachTask(backend, BimanualReachTaskCfg())
    assert task.action_dim == 24
    last_action = torch.zeros(backend.num_envs, task.action_dim)

    obs = task.observe(backend, last_action)
    assert obs.shape == (backend.num_envs, task.observation_dim)

    # shared [q + qd] = 2 * 26 joints = 52
    # per arm: 7 + 6 + 7 + 4 = 24, * 2 arms = 48
    # last action: 24
    assert task.observation_dim == 52 + 48 + 24


def test_bimanual_reset_samples_per_arm_targets_in_workspace():
    backend = _FakeBimanualBackend()
    cfg = BimanualReachTaskCfg(
        arms=[
            ArmSpec(role="arm_left", ee_frame="arm_left:tool0",
                    workspace_lo=(0.1, 0.1, 0.1), workspace_hi=(0.2, 0.2, 0.2)),
            ArmSpec(role="arm_right", ee_frame="arm_right:tool0",
                    workspace_lo=(0.3, -0.3, 0.3), workspace_hi=(0.4, -0.2, 0.4)),
        ],
    )
    task = BimanualReachTask(backend, cfg)
    ids = torch.arange(backend.num_envs, dtype=torch.long)
    task.reset(backend, ids)

    tl = task._target_pos[:, 0, :]  # left arm
    tr = task._target_pos[:, 1, :]  # right arm
    assert ((tl >= 0.1 - 1e-6) & (tl <= 0.2 + 1e-6)).all()
    assert (tr[:, 0] >= 0.3 - 1e-6).all() and (tr[:, 0] <= 0.4 + 1e-6).all()
    assert (tr[:, 1] >= -0.3 - 1e-6).all() and (tr[:, 1] <= -0.2 + 1e-6).all()


def test_bimanual_reward_sums_per_arm_distance():
    backend = _FakeBimanualBackend()
    cfg = BimanualReachTaskCfg(
        pos_weight=1.0, ori_weight=0.0,
        action_penalty=0.0, joint_vel_penalty=0.0,
    )
    task = BimanualReachTask(backend, cfg)
    # Place left target 0.1 m from left ee, right target 0.2 m from right ee.
    task._target_pos[:, 0, :] = torch.tensor([0.3, 0.1 + 0.1, 0.3])  # dist = 0.1
    task._target_pos[:, 1, :] = torch.tensor([0.3, -0.1 - 0.2, 0.3])  # dist = 0.2

    last_action = torch.zeros(backend.num_envs, task.action_dim)
    reward = task.reward(backend, last_action)

    # reward = -1.0 * (0.1 + 0.2) = -0.3
    assert torch.allclose(reward, torch.full((backend.num_envs,), -0.3), atol=1e-5)


def test_bimanual_done_requires_both_arms_in_threshold():
    backend = _FakeBimanualBackend()
    task = BimanualReachTask(backend, BimanualReachTaskCfg(
        success_pos_threshold=0.05, success_hold_steps=1,
    ))
    # Env 0: both arms on target. Env 1: only left on target. Env 2: neither.
    robot = backend.robots["robot"]
    robot._ee_poses["arm_left:tool0"] = torch.tensor([
        [0.30, 0.10, 0.30, 1.0, 0.0, 0.0, 0.0],
        [0.30, 0.10, 0.30, 1.0, 0.0, 0.0, 0.0],
        [0.50, 0.50, 0.50, 1.0, 0.0, 0.0, 0.0],
    ])
    robot._ee_poses["arm_right:tool0"] = torch.tensor([
        [0.30, -0.10, 0.30, 1.0, 0.0, 0.0, 0.0],
        [0.50, 0.50, 0.50, 1.0, 0.0, 0.0, 0.0],  # far
        [0.50, 0.50, 0.50, 1.0, 0.0, 0.0, 0.0],
    ])
    task._target_pos[:, 0, :] = torch.tensor([0.30, 0.10, 0.30])
    task._target_pos[:, 1, :] = torch.tensor([0.30, -0.10, 0.30])

    step = torch.zeros(backend.num_envs, dtype=torch.long)
    terminated, _ = task.done(backend, step, max_steps=100)
    # Only env 0 hits both.
    assert terminated.tolist() == [True, False, False]
