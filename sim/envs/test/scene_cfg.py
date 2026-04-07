from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from sim.assets import make_ar5_l6_left_robot_cfg, make_ar5_l6_right_robot_cfg, make_workspace_table_cfg


@configclass
class TestSceneCfg(InteractiveSceneCfg):
    """Minimal smoke-test scene for AR5_L6 robot + workspace table."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    robot: ArticulationCfg = make_ar5_l6_left_robot_cfg()
    table: AssetBaseCfg = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Table")

    def __post_init__(self):
        super().__post_init__()
        # Table + robot defaults live in sim.assets.scene_assets.
        self.table = make_workspace_table_cfg(prim_path="{ENV_REGEX_NS}/Table")
        self.robot = make_ar5_l6_left_robot_cfg(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class TestDualSceneCfg(InteractiveSceneCfg):
    """Smoke-test scene with both AR5_L6 left and right robots + shared table."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    robot_left: ArticulationCfg = make_ar5_l6_left_robot_cfg(prim_path="{ENV_REGEX_NS}/RobotLeft")
    robot_right: ArticulationCfg = make_ar5_l6_right_robot_cfg(prim_path="{ENV_REGEX_NS}/RobotRight")
    table: AssetBaseCfg = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Table")

    def __post_init__(self):
        super().__post_init__()
        self.table = make_workspace_table_cfg(prim_path="{ENV_REGEX_NS}/Table")
        self.robot_left = make_ar5_l6_left_robot_cfg(prim_path="{ENV_REGEX_NS}/RobotLeft")
        self.robot_right = make_ar5_l6_right_robot_cfg(prim_path="{ENV_REGEX_NS}/RobotRight")
