from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.sim.converters import MeshConverter, MeshConverterCfg


_REPO_ROOT = Path(__file__).resolve().parents[2]
_GENERATED_USD_DIR = _REPO_ROOT / "assets" / "usd_generated"

WORKSPACE_TABLE_STL_PATH = _REPO_ROOT / "assets" / "urdf" / "work_space2.stl"

WORKSPACE_TABLE_MESH_CONVERTER_CFG = MeshConverterCfg(
    asset_path=str(WORKSPACE_TABLE_STL_PATH),
    usd_dir=str(_GENERATED_USD_DIR),
    usd_file_name="work_space2.usd",
    force_usd_conversion=False,
    make_instanceable=True,
    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    rigid_props=sim_utils.RigidBodyPropertiesCfg(
        kinematic_enabled=True,
        disable_gravity=True,
    ),
)


def resolve_workspace_table_usd_path() -> str:
    """Converts table STL to USD if needed and returns the USD path."""
    converter = MeshConverter(WORKSPACE_TABLE_MESH_CONVERTER_CFG)
    return converter.usd_path


def make_workspace_table_cfg(
    prim_path: str = "{ENV_REGEX_NS}/Table",
    pos: tuple[float, float, float] = (0.5, 0.0, 0.0),
    rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> AssetBaseCfg:
    """Builds an Isaac Lab AssetBaseCfg for the workspace table mesh."""
    return AssetBaseCfg(
        prim_path=prim_path,
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos, rot=rot),
        spawn=sim_utils.UsdFileCfg(
            usd_path=resolve_workspace_table_usd_path(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
    )
