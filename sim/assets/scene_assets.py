from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg

from .robots import AR5_L6_LEFT_CFG


_REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSTATION_URDF_PATH = _REPO_ROOT / "assets" / "urdf" / "workstation.urdf"

# Workstation fixed at env origin. Quaternion is (w, x, y, z).
WORKSTATION_TABLE_POS = (0.0, 0.0, 0.0)
WORKSTATION_TABLE_ROT = (1.0, 0.0, 0.0, 0.0)

# Robot root pose in env frame, relative to workstation origin (tune here for all scenes).
ROBOT_POS_REL_TO_WORKSTATION = (0.0637, 0.715, 1.267)
ROBOT_ROT_REL_TO_WORKSTATION = (0.5, -0.5, 0.5, 0.5)


def make_ar5_l6_left_robot_cfg(
    prim_path: str = "{ENV_REGEX_NS}/Robot",
    pos: tuple[float, float, float] | None = None,
    rot: tuple[float, float, float, float] | None = None,
) -> ArticulationCfg:
    """Left arm + hand with default pose relative to the workstation (env frame)."""
    if pos is None:
        pos = ROBOT_POS_REL_TO_WORKSTATION
    if rot is None:
        rot = ROBOT_ROT_REL_TO_WORKSTATION
    return AR5_L6_LEFT_CFG.replace(
        prim_path=prim_path,
        init_state=AR5_L6_LEFT_CFG.init_state.replace(pos=pos, rot=rot),
    )


def make_workspace_table_cfg(
    prim_path: str = "{ENV_REGEX_NS}/Table",
    pos: tuple[float, float, float] | None = None,
    rot: tuple[float, float, float, float] | None = None,
) -> AssetBaseCfg:
    """Builds an Isaac Lab AssetBaseCfg for the workstation URDF.

    Defaults place the table at the env origin. Override ``pos``/``rot`` only for special layouts.
    """
    if pos is None:
        pos = WORKSTATION_TABLE_POS
    if rot is None:
        rot = WORKSTATION_TABLE_ROT
    urdf_cfg = sim_utils.UrdfFileCfg(
        asset_path=str(WORKSTATION_URDF_PATH),
        fix_base=True,
        merge_fixed_joints=False,
        make_instanceable=False,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=None,
                damping=None,
            )
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    )

    return AssetBaseCfg(
        prim_path=prim_path,
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos, rot=rot),
        spawn=urdf_cfg,
    )
