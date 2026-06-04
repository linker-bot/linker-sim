"""Pure-Python tests for the MuJoCo backend.

Skipped if `mujoco` isn't importable. Exercises the `Robot` contract on
the composed `ar5_l6_bench_bimanual` workstation and verifies a JointPD
smoke rollout holds the default pose.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
mujoco = pytest.importorskip("mujoco")

from linker_sim.backends.base import Robot
from linker_sim.backends.mujoco.backend import MujocoBackendCfg, MujocoSimBackend
from linker_sim.controllers.joint_pd import JointPDController, JointPDControllerCfg


# ar5_l6_bench_bimanual: arm_left/arm_right have 7 actuated each, hand_left/
# hand_right have 6 actuated + 5 mimic each.
# Total packed columns = 7 + 7 + (6 + 5) + (6 + 5) = 36.
ARM_N = 7
HAND_ACT = 6
HAND_MIMIC = 5
EXPECTED_NJOINTS = 2 * ARM_N + 2 * (HAND_ACT + HAND_MIMIC)


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
    assert robot.joint_ids_of("arm_left").shape == (ARM_N,)
    assert robot.actuated_joint_ids_of("arm_left").shape == (ARM_N,)
    assert robot.joint_ids_of("arm_right").shape == (ARM_N,)
    assert robot.joint_ids_of("hand_left").shape == (HAND_ACT + HAND_MIMIC,)
    assert robot.actuated_joint_ids_of("hand_left").shape == (HAND_ACT,)
    assert robot.joint_ids_of("hand_right").shape == (HAND_ACT + HAND_MIMIC,)
    assert robot.actuated_joint_ids_of("hand_right").shape == (HAND_ACT,)


def test_mass_and_jacobian_shapes(robot):
    assert robot.mass_matrix("arm_left").shape == (1, ARM_N, ARM_N)
    assert robot.jacobian("arm_left", "arm_left:tool0").shape == (1, 6, ARM_N)
    assert robot.mass_matrix("hand_left").shape == (1, HAND_ACT, HAND_ACT)
    assert robot.gravity("arm_right").shape == (1, ARM_N)


def test_frame_resolution_matches_manifest(robot):
    handle = robot.handle
    assert robot.body_id_of("arm_left:tool0") == robot.body_id_of(handle.frames["arm_left:tool0"])
    assert robot.body_id_of("arm_right:tool0") == robot.body_id_of(handle.frames["arm_right:tool0"])
    with pytest.raises(KeyError):
        robot.body_id_of("arm_left:does_not_exist")


def test_ee_pose_b_shape_and_root_identity(robot):
    pose = robot.ee_pose_b("arm_left:tool0")
    assert pose.shape == (1, 7)
    quat = pose[0, 3:7].numpy()
    assert abs(float((quat ** 2).sum()) - 1.0) < 1e-5


def test_write_joint_state_roundtrip(robot):
    target = robot.joint_pos_default.clone()
    target[0, 0] += 0.123
    target[0, 2 * ARM_N] += 0.05  # first hand_left joint after both arms
    zero_vel = torch.zeros_like(target)
    robot.write_joint_state(target, zero_vel)
    got = robot.joint_pos
    assert torch.allclose(got, target, atol=1e-6)


def test_jointpd_writes_data_ctrl(backend, robot):
    ctrl = JointPDController(JointPDControllerCfg(role="arm_left", action_scale=0.1))
    ctrl.attach(robot)
    cmd = torch.ones(1, ctrl.command_dim) * 0.5
    ctrl.set_command(cmd, robot)
    ctrl.apply(robot)
    expected = (robot.joint_pos_default[0, :ARM_N] + 0.1 * 0.5).numpy()
    arm_actuator_ids = []
    for jname in robot.handle.joints["arm_left"]:
        jid = mujoco.mj_name2id(backend._model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        for a in range(backend._model.nu):
            if int(backend._model.actuator_trnid[a, 0]) == jid:
                arm_actuator_ids.append(a)
                break
    assert len(arm_actuator_ids) == ARM_N
    for a, exp in zip(arm_actuator_ids, expected, strict=True):
        assert abs(float(backend._data.ctrl[a]) - float(exp)) < 1e-6


def test_zero_command_rollout_holds_default(backend, robot):
    arm_l = JointPDController(JointPDControllerCfg(role="arm_left", action_scale=0.0))
    arm_r = JointPDController(JointPDControllerCfg(role="arm_right", action_scale=0.0))
    hand_l = JointPDController(JointPDControllerCfg(role="hand_left", action_scale=0.0))
    hand_r = JointPDController(JointPDControllerCfg(role="hand_right", action_scale=0.0))
    for c in (arm_l, arm_r, hand_l, hand_r):
        c.attach(robot)
        c.set_command(torch.zeros(1, c.command_dim), robot)

    qpos0 = robot.joint_pos_default.clone()
    for _ in range(500):
        for c in (arm_l, arm_r, hand_l, hand_r):
            c.apply(robot)
        backend.step()
        assert torch.isfinite(robot.joint_pos).all(), "qpos went NaN/Inf"

    # Steady-state gravity sag under the manifest's PD gains.
    drift = (robot.joint_pos - qpos0).abs().max().item()
    assert drift < 0.1, f"qpos drifted {drift:.3e} rad from default"
