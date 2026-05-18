"""Pure-Python tests for the MuJoCo backend.

Skipped if `mujoco` isn't importable. Exercises the `Robot` contract on
the composed `ar5_l6_bench` workstation and verifies a JointPD smoke
rollout holds the default pose.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
mujoco = pytest.importorskip("mujoco")

from sim.backends.base import Robot
from sim.backends.mujoco.backend import MujocoBackendCfg, MujocoSimBackend
from sim.controllers.joint_pd import JointPDController, JointPDControllerCfg


# ar5_l6_bench: arm has 7 actuated + 0 mimic, hand has 6 actuated + 5 mimic.
# Total packed columns = 7 + 6 + 5 = 18.
EXPECTED_NJOINTS = 18
ARM_N = 7
HAND_ACT = 6
HAND_MIMIC = 5


@pytest.fixture(scope="module")
def backend():
    be = MujocoSimBackend(MujocoBackendCfg())
    yield be
    be.close()


@pytest.fixture()
def robot(backend):
    backend.reset()
    return backend.robots["robot"]


def test_robot_satisfies_protocol(robot):
    assert isinstance(robot, Robot)


def test_joint_counts_and_shapes(robot):
    assert robot.joint_pos.shape == (1, EXPECTED_NJOINTS)
    assert robot.joint_vel.shape == (1, EXPECTED_NJOINTS)
    assert robot.joint_pos_default.shape == (1, EXPECTED_NJOINTS)
    assert robot.joint_ids_of("arm").shape == (ARM_N,)
    assert robot.actuated_joint_ids_of("arm").shape == (ARM_N,)
    assert robot.joint_ids_of("hand").shape == (HAND_ACT + HAND_MIMIC,)
    assert robot.actuated_joint_ids_of("hand").shape == (HAND_ACT,)


def test_mass_and_jacobian_shapes(robot):
    assert robot.mass_matrix("arm").shape == (1, ARM_N, ARM_N)
    assert robot.jacobian("arm", "arm:tool0").shape == (1, 6, ARM_N)
    assert robot.mass_matrix("hand").shape == (1, HAND_ACT, HAND_ACT)
    assert robot.gravity("arm").shape == (1, ARM_N)


def test_frame_resolution_matches_manifest(robot):
    handle = robot.handle
    assert robot.body_id_of(handle.ee_link) > 0
    assert robot.body_id_of("arm:tool0") == robot.body_id_of(handle.frames["arm:tool0"])
    with pytest.raises(KeyError):
        robot.body_id_of("arm:does_not_exist")


def test_ee_pose_b_shape_and_root_identity(robot):
    pose = robot.ee_pose_b("arm:tool0")
    assert pose.shape == (1, 7)
    quat = pose[0, 3:7].numpy()
    assert abs(float((quat ** 2).sum()) - 1.0) < 1e-5


def test_write_joint_state_roundtrip(robot):
    target = robot.joint_pos_default.clone()
    target[0, 0] += 0.123
    target[0, ARM_N] += 0.05  # first hand joint
    zero_vel = torch.zeros_like(target)
    robot.write_joint_state(target, zero_vel)
    got = robot.joint_pos
    assert torch.allclose(got, target, atol=1e-6)


def test_jointpd_writes_data_ctrl(backend, robot):
    ctrl = JointPDController(JointPDControllerCfg(role="arm", action_scale=0.1))
    ctrl.attach(robot)
    cmd = torch.ones(1, ctrl.command_dim) * 0.5
    ctrl.set_command(cmd, robot)
    ctrl.apply(robot)
    expected = (robot.joint_pos_default[0, :ARM_N] + 0.1 * 0.5).numpy()
    arm_actuator_ids = []
    for jname in robot.handle.joints["arm"]:
        jid = mujoco.mj_name2id(backend._model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        for a in range(backend._model.nu):
            if int(backend._model.actuator_trnid[a, 0]) == jid:
                arm_actuator_ids.append(a)
                break
    assert len(arm_actuator_ids) == ARM_N
    for a, exp in zip(arm_actuator_ids, expected, strict=True):
        assert abs(float(backend._data.ctrl[a]) - float(exp)) < 1e-6


def test_zero_command_rollout_holds_default(backend, robot):
    arm_ctrl = JointPDController(JointPDControllerCfg(role="arm", action_scale=0.0))
    hand_ctrl = JointPDController(JointPDControllerCfg(role="hand", action_scale=0.0))
    arm_ctrl.attach(robot)
    hand_ctrl.attach(robot)
    zero_arm = torch.zeros(1, arm_ctrl.command_dim)
    zero_hand = torch.zeros(1, hand_ctrl.command_dim)
    arm_ctrl.set_command(zero_arm, robot)
    hand_ctrl.set_command(zero_hand, robot)

    qpos0 = robot.joint_pos_default.clone()
    for _ in range(500):
        arm_ctrl.apply(robot)
        hand_ctrl.apply(robot)
        backend.step()
        assert torch.isfinite(robot.joint_pos).all(), "qpos went NaN/Inf"

    # Steady-state gravity sag under the manifest's `arm` PD (k=1000, d=4).
    # The per-component GRAV_HOLD check disables gravity & contact and asks
    # for sub-mrad drift; this runtime test runs full physics, so we only
    # require the arm hasn't fallen out of its pose envelope (~6 deg).
    drift = (robot.joint_pos - qpos0).abs().max().item()
    assert drift < 0.1, f"qpos drifted {drift:.3e} rad from default"
