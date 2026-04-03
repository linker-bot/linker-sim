from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


_REPO_ROOT = Path(__file__).resolve().parents[2]
AR5_L6_LEFT_URDF_PATH = _REPO_ROOT / "assets" / "urdf" / "AR5_L6_description" / "AR5_L6_left.urdf"


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
            stiffness=80.0,
            damping=4.0,
        ),
        "hand": ImplicitActuatorCfg(
            joint_names_expr=["lh_.*"],
            stiffness=20.0,
            damping=2.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""AR5 + L6 left robot loaded directly from URDF."""
