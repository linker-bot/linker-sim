"""OSC (operational-space control) controller.

Wraps IsaacLab's `OperationalSpaceController` (battle-tested math) but
exposes it through the sim-agnostic `Controller` protocol. Command
semantics mirror the legacy `TestOscRLEnv`:

- Input `(B, 6)` is a delta-pose in the root frame, clipped to `[-1, 1]`
  by the env then scaled per-axis (`arm_action_scale_{pos,rot}`) here.
- `apply()` reads Jacobian / mass matrix / gravity from the robot for
  the controller's role and writes joint effort.
- Null-space target is the default joint pose.

On `attach()` the controller overrides the arm role's actuator gains to
the OSC profile (low stiffness, tuned damping) so the effort-based
output dominates over the implicit PD drive. This matches the legacy
`control_mode="osc"` branch in `sim/backends/isaac/loaders.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from isaaclab.controllers import OperationalSpaceController, OperationalSpaceControllerCfg

from sim.backends.base import Robot
from sim.backends.isaac.robot import IsaacRobot


@dataclass
class OscControllerCfg:
    role: str = "arm"
    frame: str | None = None  # None -> handle.ee_link

    action_scale_pos: float = 0.05
    action_scale_rot: float = 0.25

    # Profile applied on attach(). Matches the legacy OSC config.
    stiffness: tuple[float, ...] = (150.0, 150.0, 150.0, 80.0, 80.0, 80.0)
    damping_ratio: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    nullspace_stiffness: float = 10.0
    nullspace_damping_ratio: float = 1.0

    # Actuator-gain override (PD drive lowered so effort dominates).
    actuator_stiffness: float = 150.0
    actuator_damping: float = 8.0


class OscController:
    """Isaac-backed OSC controller.

    Keeps a reference to the underlying IsaacLab `OperationalSpaceController`
    and routes state through the `Robot` protocol.

    Depends on Isaac-specific internals only via `isaaclab.controllers`;
    a MuJoCo-side `OscController` will reimplement the math natively
    when PR #1b unblocks the MuJoCo backend.
    """

    role: str
    command_dim: int = 6

    def __init__(self, cfg: OscControllerCfg | None = None):
        self.cfg = cfg or OscControllerCfg()
        self.role = self.cfg.role
        self._controller: OperationalSpaceController | None = None
        self._last_command: torch.Tensor | None = None  # (B, 6), already scaled

    # -- Controller interface -------------------------------------------- #

    def attach(self, robot: Robot) -> None:
        if not isinstance(robot, IsaacRobot):
            raise TypeError(
                "OscController currently only supports IsaacRobot. "
                "MuJoCo-side OSC lands with PR #1b."
            )

        ctrl_cfg = OperationalSpaceControllerCfg(
            target_types=["pose_rel"],
            motion_control_axes_task=(1, 1, 1, 1, 1, 1),
            inertial_dynamics_decoupling=True,
            partial_inertial_dynamics_decoupling=False,
            gravity_compensation=True,
            impedance_mode="fixed",
            motion_stiffness_task=self.cfg.stiffness,
            motion_damping_ratio_task=self.cfg.damping_ratio,
            nullspace_control="position",
            nullspace_stiffness=self.cfg.nullspace_stiffness,
            nullspace_damping_ratio=self.cfg.nullspace_damping_ratio,
        )
        self._controller = OperationalSpaceController(
            ctrl_cfg, robot.num_envs, str(robot.device)
        )
        # Lower the actuator PD so effort dominates.
        robot.write_gains(
            self.role,
            stiffness=self.cfg.actuator_stiffness,
            damping=self.cfg.actuator_damping,
        )
        self._last_command = torch.zeros((robot.num_envs, 6), device=robot.device)

    def set_command(self, command: torch.Tensor, robot: Robot) -> None:
        # Command is (B, 6) delta pose in [-1, 1]; scale per-axis.
        scaled = command.clone()
        scaled[:, 0:3] *= self.cfg.action_scale_pos
        scaled[:, 3:6] *= self.cfg.action_scale_rot
        self._last_command = scaled

        ee_pose_b = robot.ee_pose_b(self.cfg.frame)
        assert self._controller is not None
        self._controller.set_command(command=scaled, current_ee_pose_b=ee_pose_b)

    def apply(self, robot: Robot) -> None:
        assert self._controller is not None
        joint_ids = robot.actuated_joint_ids_of(self.role)  # type: ignore[attr-defined]

        jacobian = robot.jacobian(self.role, self.cfg.frame)
        mass_matrix = robot.mass_matrix(self.role)
        gravity = robot.gravity(self.role)
        ee_pose_b = robot.ee_pose_b(self.cfg.frame)
        ee_vel_b = robot.ee_vel_b(self.cfg.frame)
        joint_pos = robot.joint_pos[:, joint_ids]
        joint_vel = robot.joint_vel[:, joint_ids]
        null_target = robot.joint_pos_default[:, joint_ids]

        efforts = self._controller.compute(
            jacobian_b=jacobian,
            current_ee_pose_b=ee_pose_b,
            current_ee_vel_b=ee_vel_b,
            mass_matrix=mass_matrix,
            gravity=gravity,
            current_joint_pos=joint_pos,
            current_joint_vel=joint_vel,
            nullspace_joint_pos_target=null_target,
        )
        robot.set_joint_effort(efforts, joint_ids=joint_ids)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if self._controller is not None:
            self._controller.reset()
        if self._last_command is not None:
            if env_ids is None:
                self._last_command.zero_()
            else:
                self._last_command[env_ids] = 0.0
