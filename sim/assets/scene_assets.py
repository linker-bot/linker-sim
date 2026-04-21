from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg

from .robots import AR5_L6_LEFT_CFG, AR5_L6_LEFT_OSC_CFG, AR5_L6_RIGHT_CFG, AR5_L6_RIGHT_OSC_CFG


_REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSTATION_URDF_PATH = _REPO_ROOT / "assets" / "urdf" / "workstation.urdf"

# Workstation fixed at env origin. Quaternion is (w, x, y, z).
WORKSTATION_TABLE_POS = (0.0, 0.0, 0.0)
WORKSTATION_TABLE_ROT = (1.0, 0.0, 0.0, 0.0)

# Robot root pose in env frame, relative to workstation origin.
# Tune left/right independently.
ROBOT_LEFT_POS_REL_TO_WORKSTATION = (0.0637, 0.719, 1.267)
ROBOT_LEFT_ROT_REL_TO_WORKSTATION = (0.5, -0.5, 0.5, 0.5)
ROBOT_RIGHT_POS_REL_TO_WORKSTATION = (0.0637, 0.536, 1.267)
ROBOT_RIGHT_ROT_REL_TO_WORKSTATION = (0.5, 0.5, 0.5, -0.5)

# Backward-compatible aliases (existing code may still import these).
ROBOT_POS_REL_TO_WORKSTATION = ROBOT_LEFT_POS_REL_TO_WORKSTATION
ROBOT_ROT_REL_TO_WORKSTATION = ROBOT_LEFT_ROT_REL_TO_WORKSTATION


def get_robot_default_pose(side: str = "left") -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Return default (pos, rot-wxyz) for the requested robot side."""
    side_norm = side.lower()
    if side_norm == "left":
        return ROBOT_LEFT_POS_REL_TO_WORKSTATION, ROBOT_LEFT_ROT_REL_TO_WORKSTATION
    if side_norm == "right":
        return ROBOT_RIGHT_POS_REL_TO_WORKSTATION, ROBOT_RIGHT_ROT_REL_TO_WORKSTATION
    raise ValueError(f"Unknown robot side: {side!r}. Expected 'left' or 'right'.")


def make_ar5_l6_left_robot_cfg(
    prim_path: str = "{ENV_REGEX_NS}/Robot",
    pos: tuple[float, float, float] | None = None,
    rot: tuple[float, float, float, float] | None = None,
    control_mode: str = "joint",
) -> ArticulationCfg:
    """Left arm + hand with default pose relative to the workstation (env frame)."""
    default_pos, default_rot = get_robot_default_pose("left")
    if pos is None:
        pos = default_pos
    if rot is None:
        rot = default_rot
    base_cfg = AR5_L6_LEFT_OSC_CFG if control_mode.lower() == "osc" else AR5_L6_LEFT_CFG
    return base_cfg.replace(
        prim_path=prim_path,
        init_state=base_cfg.init_state.replace(pos=pos, rot=rot),
    )


def make_ar5_l6_right_robot_cfg(
    prim_path: str = "{ENV_REGEX_NS}/Robot",
    pos: tuple[float, float, float] | None = None,
    rot: tuple[float, float, float, float] | None = None,
    control_mode: str = "joint",
) -> ArticulationCfg:
    """Right arm + hand with default pose relative to the workstation (env frame)."""
    default_pos, default_rot = get_robot_default_pose("right")
    if pos is None:
        pos = default_pos
    if rot is None:
        rot = default_rot
    base_cfg = AR5_L6_RIGHT_OSC_CFG if control_mode.lower() == "osc" else AR5_L6_RIGHT_CFG
    return base_cfg.replace(
        prim_path=prim_path,
        init_state=base_cfg.init_state.replace(pos=pos, rot=rot),
    )


def make_ar5_l6_robot_cfg(
    side: str = "left",
    prim_path: str = "{ENV_REGEX_NS}/Robot",
    pos: tuple[float, float, float] | None = None,
    rot: tuple[float, float, float, float] | None = None,
    control_mode: str = "joint",
) -> ArticulationCfg:
    """Build left/right AR5_L6 robot cfg from a common entrypoint."""
    side_norm = side.lower()
    if side_norm == "left":
        return make_ar5_l6_left_robot_cfg(prim_path=prim_path, pos=pos, rot=rot, control_mode=control_mode)
    if side_norm == "right":
        return make_ar5_l6_right_robot_cfg(prim_path=prim_path, pos=pos, rot=rot, control_mode=control_mode)
    raise ValueError(f"Unknown robot side: {side!r}. Expected 'left' or 'right'.")


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
        # Keep conversion behavior aligned with robot importer settings.
        merge_fixed_joints=True,
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
