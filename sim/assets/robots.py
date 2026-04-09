from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


_REPO_ROOT = Path(__file__).resolve().parents[2]
AR5_L6_LEFT_URDF_PATH = _REPO_ROOT / "assets" / "urdf" / "AR5_L6_description" / "AR5_L6_left.urdf"
AR5_L6_RIGHT_URDF_PATH = _REPO_ROOT / "assets" / "urdf" / "AR5_L6_description" / "AR5_L6_right.urdf"

# Implicit actuator PD gains (PhysX joint drives), now per-joint arrays
# to match the simtoolreal-style tuning workflow.
# Tune with:
#   python sim/envs/test/gain_tuner_scene.py --num_envs 1 --robot_side left
# Then persist values in this file (and/or sim/envs/test/joint_gains.json).

AR5_L6_LEFT_ARM_JOINTS = [f"AR5_5_07L_W4C4A2_joint_{i}" for i in range(1, 8)]
AR5_L6_LEFT_HAND_JOINTS = [
    "lh_thumb_cmc_roll",
    "lh_thumb_cmc_pitch",
    "lh_thumb_dip",
    "lh_index_mcp_pitch",
    "lh_index_dip",
    "lh_middle_mcp_pitch",
    "lh_middle_dip",
    "lh_ring_mcp_pitch",
    "lh_ring_dip",
    "lh_pinky_mcp_pitch",
    "lh_pinky_dip",
]

AR5_L6_RIGHT_ARM_JOINTS = [f"AR5_5_07R_W4C4A2_joint_{i}" for i in range(1, 8)]
AR5_L6_RIGHT_HAND_JOINTS = [
    "rh_thumb_cmc_roll",
    "rh_thumb_cmc_pitch",
    "rh_thumb_dip",
    "rh_index_mcp_pitch",
    "rh_index_dip",
    "rh_middle_mcp_pitch",
    "rh_middle_dip",
    "rh_ring_mcp_pitch",
    "rh_ring_dip",
    "rh_pinky_mcp_pitch",
    "rh_pinky_dip",
]

AR5_L6_LEFT_ARM_STIFFNESS = [1000.0] * len(AR5_L6_LEFT_ARM_JOINTS)
AR5_L6_LEFT_ARM_DAMPING = [4.0] * len(AR5_L6_LEFT_ARM_JOINTS)
AR5_L6_LEFT_HAND_STIFFNESS = [20.0] * len(AR5_L6_LEFT_HAND_JOINTS)
AR5_L6_LEFT_HAND_DAMPING = [2.0] * len(AR5_L6_LEFT_HAND_JOINTS)

AR5_L6_RIGHT_ARM_STIFFNESS = [80.0] * len(AR5_L6_RIGHT_ARM_JOINTS)
AR5_L6_RIGHT_ARM_DAMPING = [4.0] * len(AR5_L6_RIGHT_ARM_JOINTS)
AR5_L6_RIGHT_HAND_STIFFNESS = [20.0] * len(AR5_L6_RIGHT_HAND_JOINTS)
AR5_L6_RIGHT_HAND_DAMPING = [2.0] * len(AR5_L6_RIGHT_HAND_JOINTS)


def _joint_gain_map(joints: list[str], gains: list[float]) -> dict[str, float]:
    assert len(joints) == len(gains), f"len(joints)={len(joints)} != len(gains)={len(gains)}"
    return {joint: float(gain) for joint, gain in zip(joints, gains, strict=True)}


AR5_L6_LEFT_ARM_STIFFNESS_MAP = _joint_gain_map(AR5_L6_LEFT_ARM_JOINTS, AR5_L6_LEFT_ARM_STIFFNESS)
AR5_L6_LEFT_ARM_DAMPING_MAP = _joint_gain_map(AR5_L6_LEFT_ARM_JOINTS, AR5_L6_LEFT_ARM_DAMPING)
AR5_L6_LEFT_HAND_STIFFNESS_MAP = _joint_gain_map(AR5_L6_LEFT_HAND_JOINTS, AR5_L6_LEFT_HAND_STIFFNESS)
AR5_L6_LEFT_HAND_DAMPING_MAP = _joint_gain_map(AR5_L6_LEFT_HAND_JOINTS, AR5_L6_LEFT_HAND_DAMPING)

AR5_L6_RIGHT_ARM_STIFFNESS_MAP = _joint_gain_map(AR5_L6_RIGHT_ARM_JOINTS, AR5_L6_RIGHT_ARM_STIFFNESS)
AR5_L6_RIGHT_ARM_DAMPING_MAP = _joint_gain_map(AR5_L6_RIGHT_ARM_JOINTS, AR5_L6_RIGHT_ARM_DAMPING)
AR5_L6_RIGHT_HAND_STIFFNESS_MAP = _joint_gain_map(AR5_L6_RIGHT_HAND_JOINTS, AR5_L6_RIGHT_HAND_STIFFNESS)
AR5_L6_RIGHT_HAND_DAMPING_MAP = _joint_gain_map(AR5_L6_RIGHT_HAND_JOINTS, AR5_L6_RIGHT_HAND_DAMPING)


AR5_L6_LEFT_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(AR5_L6_LEFT_URDF_PATH),
        fix_base=True,
        merge_fixed_joints=False,
        make_instanceable=False,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "AR5_5_07L_W4C4A2_joint_[1-7]": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["AR5_5_07L_W4C4A2_joint_[1-7]"],
            stiffness=AR5_L6_LEFT_ARM_STIFFNESS_MAP,
            damping=AR5_L6_LEFT_ARM_DAMPING_MAP,
        ),
        "hand": ImplicitActuatorCfg(
            joint_names_expr=["lh_.*"],
            stiffness=AR5_L6_LEFT_HAND_STIFFNESS_MAP,
            damping=AR5_L6_LEFT_HAND_DAMPING_MAP,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""AR5 + L6 left robot loaded directly from URDF."""


AR5_L6_RIGHT_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(AR5_L6_RIGHT_URDF_PATH),
        fix_base=True,
        merge_fixed_joints=False,
        make_instanceable=False,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "AR5_5_07R_W4C4A2_joint_[1-7]": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["AR5_5_07R_W4C4A2_joint_[1-7]"],
            stiffness=AR5_L6_RIGHT_ARM_STIFFNESS_MAP,
            damping=AR5_L6_RIGHT_ARM_DAMPING_MAP,
        ),
        "hand": ImplicitActuatorCfg(
            joint_names_expr=["rh_.*"],
            stiffness=AR5_L6_RIGHT_HAND_STIFFNESS_MAP,
            damping=AR5_L6_RIGHT_HAND_DAMPING_MAP,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""AR5 + L6 right robot loaded directly from URDF."""
