"""Isaac Sim / IsaacLab asset loaders.

Converts sim-agnostic `WorkstationHandle`s from `sim.registry` into
IsaacLab-native cfgs (`ArticulationCfg`). The registry stays pure (no
Isaac imports); everything Isaac-specific lives here.

This is the only module in PR #1a that imports `isaaclab`. PR #2 adds a
matching `sim/backends/mujoco/loaders.py` once component MJCFs are
authored (see docs/component_mjcf_authoring.md).
"""

from __future__ import annotations

from typing import Literal

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from sim.registry import WorkstationHandle


# Gains used when `control_mode="osc"` is selected. Applied to the `arm`
# role only — all other roles keep the handle's default gains.
# TODO(PR #2): move to component `meta.yaml` as a named `gain_profiles`
# section; let recipes select a profile. Hardcoded here for parity with
# the legacy `AR5_L6_*_OSC_CFG` values in sim/assets/robots.py.
_OSC_ARM_STIFFNESS = 150.0
_OSC_ARM_DAMPING = 8.0


def to_articulation_cfg(
    handle: WorkstationHandle,
    *,
    prim_path: str = "{ENV_REGEX_NS}/Robot",
    control_mode: Literal["joint", "osc"] = "joint",
) -> ArticulationCfg:
    """Build an `ArticulationCfg` from a composed workstation handle.

    One `ImplicitActuatorCfg` is created per role that declares
    `default_gains` in its component meta. Both actuated and mimic
    joints are included under the same role (IsaacLab still needs a
    drive on mimic joints even though URDF `<mimic>` enforces the
    coupling).

    Parameters
    ----------
    handle:
        Return value of `sim.registry.load(name)`.
    prim_path:
        USD prim path for the spawned articulation. Usually
        `{ENV_REGEX_NS}/<Name>`.
    control_mode:
        `"joint"` keeps the component-declared default gains (high
        stiffness for direct joint-space control). `"osc"` swaps the
        arm role to the OSC profile (lower stiffness, applied on top
        of damping).
    """
    actuators: dict[str, ImplicitActuatorCfg] = {}
    for role, actuated in handle.joints.items():
        mimic = handle.mimic_joints.get(role, [])
        joint_list = list(actuated) + list(mimic)
        if not joint_list:
            continue

        gains = handle.default_gains.get(role)
        if gains is None:
            # Role has joints but no declared gains — skip the actuator
            # group. Isaac will fall back to its built-in PD defaults;
            # the component's meta.yaml should declare default_gains to
            # avoid this.
            continue

        kp = gains.stiffness
        kd = gains.damping
        if control_mode == "osc" and role == "arm":
            kp = _OSC_ARM_STIFFNESS
            kd = _OSC_ARM_DAMPING

        actuators[role] = ImplicitActuatorCfg(
            joint_names_expr=list(joint_list),
            stiffness={j: float(kp) for j in joint_list},
            damping={j: float(kd) for j in joint_list},
        )

    return ArticulationCfg(
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(handle.urdf_path),
            fix_base=True,
            merge_fixed_joints=False,
            make_instanceable=False,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=None, damping=None
                )
            ),
        ),
        prim_path=prim_path,
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={".*": 0.0},
        ),
        actuators=actuators,
        soft_joint_pos_limit_factor=1.0,
    )
