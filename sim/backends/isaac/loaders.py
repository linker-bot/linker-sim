"""Isaac Sim / IsaacLab asset loaders.

Converts sim-agnostic `WorkstationHandle`s from `sim.registry` into
IsaacLab-native cfgs (`ArticulationCfg`). The registry stays pure (no
Isaac imports); everything Isaac-specific lives here.

PR #2a change: the OSC gain override that used to live here
(`control_mode="osc" -> kp=150, kd=8 on the arm role`) is gone.
`OscController.attach()` now writes the gains at runtime via
`robot.write_gains(...)`, so the `control_mode` kwarg is deprecated
and no-ops with a DeprecationWarning. It will be removed in PR #2b.
"""

from __future__ import annotations

import warnings
from typing import Literal

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from sim.registry import WorkstationHandle


def to_articulation_cfg(
    handle: WorkstationHandle,
    *,
    prim_path: str = "{ENV_REGEX_NS}/Robot",
    control_mode: Literal["joint", "osc"] | None = None,
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
        Deprecated. Previously swapped in a lower-stiffness OSC profile
        for the arm role. Controllers now own their gain profile and
        apply it on attach via `robot.write_gains(...)`. Passing this
        arg emits a DeprecationWarning and has no effect.
    """
    if control_mode is not None:
        warnings.warn(
            "to_articulation_cfg(control_mode=...) is deprecated and has no "
            "effect. Gains are now applied by the controller on attach "
            "(sim.controllers.OscController.attach -> robot.write_gains). "
            "The argument will be removed in PR #2b.",
            DeprecationWarning,
            stacklevel=2,
        )

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
            # the component's meta.yaml should declare default_gains.
            continue

        kp = gains.stiffness
        kd = gains.damping

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
            # Decompose concave collision meshes into multiple convex pieces
            # via VHACD. The default "convex_hull" wraps the whole STL in a
            # single envelope, which for the bench table produces an invisible
            # bulge above the visible tabletop (cube spawns then bounce off the
            # bulge and slide off). Decomposition tracks the real geometry.
            collider_type="convex_decomposition",
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
