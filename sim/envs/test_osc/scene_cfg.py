"""OSC test scene configurations.

PR #1a rewires this to spawn a composed workstation as a single
articulation via the registry + Isaac loader. The legacy dual-articulation
scene (separate robot + workstation table) is retained behind
`LegacyTestOscSceneCfg` for transition; delete once callers migrate.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from sim.assets import make_ar5_l6_left_robot_cfg, make_ar5_l6_right_robot_cfg, make_workspace_table_cfg
from sim.backends.isaac.loaders import to_articulation_cfg
from sim.registry import load as load_workstation


# -------------------- new: composed workstation scene --------------------- #


def _workstation_robot_cfg(workstation_name: str, prim_path: str, control_mode: str = "osc") -> ArticulationCfg:
    """Load a composed workstation via the registry and build an Isaac cfg.

    The composed URDF bakes the table, arm, and hand into one articulation
    rooted at world, so the scene doesn't need a separate table asset.
    """
    handle = load_workstation(workstation_name)
    return to_articulation_cfg(
        handle,
        prim_path=prim_path,
        control_mode=control_mode,  # type: ignore[arg-type]
    )


@configclass
class OscWorkstationSceneCfg(InteractiveSceneCfg):
    """OSC scene for a composed workstation (table + arm + hand as one articulation).

    Use `scene_cfg.workstation_name = "<name>"` to point at a different
    workstation (defaults to `ar5_l6_bench`, the left-arm variant). The
    spawner in `__post_init__` rebuilds the robot cfg from that name.
    """

    workstation_name: str = "ar5_l6_bench"
    control_mode: str = "osc"

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    # Resolved in __post_init__ — this placeholder makes configclass happy.
    robot: ArticulationCfg = _workstation_robot_cfg(
        workstation_name="ar5_l6_bench",
        prim_path="{ENV_REGEX_NS}/Robot",
        control_mode="osc",
    )

    def __post_init__(self):
        super().__post_init__()
        self.robot = _workstation_robot_cfg(
            workstation_name=self.workstation_name,
            prim_path="{ENV_REGEX_NS}/Robot",
            control_mode=self.control_mode,
        )


# -------------------- legacy: kept for transition ------------------------- #


@configclass
class TestOscSceneCfg(InteractiveSceneCfg):
    """Legacy OSC scene (separate robot + table). Superseded by OscWorkstationSceneCfg."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    robot: ArticulationCfg = make_ar5_l6_left_robot_cfg(control_mode="osc")
    table: AssetBaseCfg = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Table")

    def __post_init__(self):
        super().__post_init__()
        self.table = make_workspace_table_cfg(prim_path="{ENV_REGEX_NS}/Table")
        self.robot = make_ar5_l6_left_robot_cfg(prim_path="{ENV_REGEX_NS}/Robot", control_mode="osc")


@configclass
class TestOscDualSceneCfg(InteractiveSceneCfg):
    """Legacy dual-arm OSC scene. Superseded by a bimanual workstation recipe (see PR1_PROGRESS.md)."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    robot_left: ArticulationCfg = make_ar5_l6_left_robot_cfg(prim_path="{ENV_REGEX_NS}/RobotLeft", control_mode="osc")
    robot_right: ArticulationCfg = make_ar5_l6_right_robot_cfg(
        prim_path="{ENV_REGEX_NS}/RobotRight", control_mode="osc"
    )
    table: AssetBaseCfg = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Table")

    def __post_init__(self):
        super().__post_init__()
        self.table = make_workspace_table_cfg(prim_path="{ENV_REGEX_NS}/Table")
        self.robot_left = make_ar5_l6_left_robot_cfg(prim_path="{ENV_REGEX_NS}/RobotLeft", control_mode="osc")
        self.robot_right = make_ar5_l6_right_robot_cfg(prim_path="{ENV_REGEX_NS}/RobotRight", control_mode="osc")
