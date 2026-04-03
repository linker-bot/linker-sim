from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from sim.assets import AR5_L6_LEFT_CFG, make_workspace_table_cfg


@configclass
class TestSceneCfg(InteractiveSceneCfg):
    """Minimal smoke-test scene for the AR5_L6 left robot and workspace table."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    robot: ArticulationCfg = AR5_L6_LEFT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    table: AssetBaseCfg = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Table")

    def __post_init__(self):
        super().__post_init__()
        # Table spawn is resolved here so mesh conversion happens only when constructing the scene.
        self.table = make_workspace_table_cfg(
            prim_path="{ENV_REGEX_NS}/Table",
            pos=(0.5, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        )
