"""OSC RL env over a composed workstation.

PR #1a rewires this to drive the composed workstation via the registry
instead of the legacy `make_ar5_l6_robot_cfg` path. Joint and body names
come from `sim.registry.load(cfg.workstation_name)` — no hardcoded
regexes. One articulation per env; `workstation_name` selects left vs
right variant (or a future bimanual variant).

Legacy `--robot_side both` (dual articulation) is temporarily unsupported
and will return with a bimanual workstation recipe. See PR1_PROGRESS.md.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.controllers import OperationalSpaceController, OperationalSpaceControllerCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from sim.envs.test_osc.scene_cfg import OscWorkstationSceneCfg
from sim.registry import WorkstationHandle, load as load_workstation


@configclass
class TestOscRLEnvCfg(DirectRLEnvCfg):
    """Minimal OSC-first RL config using EEF pose deltas + hand action channel."""

    decimation: int = 4
    episode_length_s: float = 8.0
    action_space: int = 1
    observation_space: int = 1

    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, render_interval=4)
    scene: InteractiveSceneCfg = OscWorkstationSceneCfg(num_envs=64, env_spacing=2.5)

    # Name of the workstation to spawn. Must exist under
    # `assets/workstations/<name>/` with a composed `workstation.urdf`.
    workstation_name: str = "ar5_l6_bench"

    ee_frame: Literal["tcp", "wrist"] = "tcp"
    arm_action_scale_pos: float = 0.05
    arm_action_scale_rot: float = 0.25
    hand_action_scale: float = 0.2
    reset_joint_noise_scale: float = 0.02

    action_penalty_scale: float = 0.01
    joint_vel_penalty_scale: float = 0.001
    pose_error_penalty_scale: float = 1.0

    osc_stiffness: tuple[float, float, float, float, float, float] = (150.0, 150.0, 150.0, 80.0, 80.0, 80.0)
    osc_damping_ratio: tuple[float, float, float, float, float, float] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    osc_nullspace_stiffness: float = 10.0


class TestOscRLEnv(DirectRLEnv):
    """OSC control env where actions are EEF deltas and hand commands."""

    cfg: TestOscRLEnvCfg

    def __init__(self, cfg: TestOscRLEnvCfg, render_mode: str | None = None, **kwargs):
        cfg.scene = OscWorkstationSceneCfg(
            num_envs=cfg.scene.num_envs,
            env_spacing=cfg.scene.env_spacing,
            workstation_name=cfg.workstation_name,
            control_mode="osc",
        )
        self._robot_names = ["robot"]
        self._handle: WorkstationHandle = load_workstation(cfg.workstation_name)
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

    def _setup_scene(self):
        pass

    def _configure_gym_env_spaces(self):
        handle = self._handle
        self._robots = [self.scene[name] for name in self._robot_names]
        self._arm_joint_ids: list[torch.Tensor] = []
        self._hand_joint_ids: list[torch.Tensor] = []
        self._ee_body_ids: list[int] = []
        self._jacobi_body_ids: list[int] = []
        self._osc_ctrls: list[OperationalSpaceController] = []
        self._arm_action_dims: list[int] = []
        self._hand_action_dims: list[int] = []
        self._joint_dims: list[int] = []

        # Joint and body names come from the workstation manifest. Arm
        # ordering follows the component's meta.actuated_joints (composer
        # preserves it in manifest.joints). Hand joint list includes mimic
        # joints so Isaac's actuator drives apply to all 11; the URDF
        # <mimic> tag constrains the 5 coupled ones at solve time.
        arm_joint_names = list(handle.joints.get("arm", []))
        hand_joint_names = list(handle.joints.get("hand", [])) + list(
            handle.mimic_joints.get("hand", [])
        )
        if not arm_joint_names:
            raise ValueError(
                f"workstation {handle.name!r} has no 'arm' role — cannot run "
                f"TestOscRLEnv against it (need at least one arm role)"
            )

        # EE frame selection. Manifest's ee_link is the arm's tool0 (TCP);
        # the legacy env also supported a "wrist" mode using link7 for
        # comparison. Derive the wrist link from the ee link by stripping
        # the _tcp suffix and appending _link7 — works for AR5 naming,
        # error-surfaced otherwise.
        if self.cfg.ee_frame == "tcp":
            ee_name = handle.ee_link
        elif self.cfg.ee_frame == "wrist":
            ee_name = _tcp_to_wrist(handle.ee_link)
        else:
            raise ValueError(f"cfg.ee_frame must be 'tcp' or 'wrist', got {self.cfg.ee_frame!r}")

        for robot in self._robots:
            arm_ids, _ = robot.find_joints(arm_joint_names)
            hand_ids, _ = robot.find_joints(hand_joint_names)
            body_ids, _ = robot.find_bodies(ee_name)
            if len(body_ids) != 1:
                raise ValueError(f"Expected one body for ee frame '{ee_name}', found {len(body_ids)}")

            ee_body_id = int(body_ids[0])
            jacobi_body_id = ee_body_id - 1 if robot.is_fixed_base else ee_body_id
            self._arm_joint_ids.append(torch.tensor(arm_ids, device=self.device, dtype=torch.long))
            self._hand_joint_ids.append(torch.tensor(hand_ids, device=self.device, dtype=torch.long))
            self._ee_body_ids.append(ee_body_id)
            self._jacobi_body_ids.append(jacobi_body_id)
            self._arm_action_dims.append(6)
            self._hand_action_dims.append(len(hand_ids))
            self._joint_dims.append(robot.data.default_joint_pos.shape[1])

            osc_cfg = OperationalSpaceControllerCfg(
                target_types=["pose_rel"],
                motion_control_axes_task=(1, 1, 1, 1, 1, 1),
                inertial_dynamics_decoupling=True,
                partial_inertial_dynamics_decoupling=False,
                gravity_compensation=True,
                impedance_mode="fixed",
                motion_stiffness_task=self.cfg.osc_stiffness,
                motion_damping_ratio_task=self.cfg.osc_damping_ratio,
                nullspace_control="position",
                nullspace_stiffness=self.cfg.osc_nullspace_stiffness,
                nullspace_damping_ratio=1.0,
            )
            self._osc_ctrls.append(OperationalSpaceController(osc_cfg, self.num_envs, str(self.device)))

        self._action_dim = int(sum(a + h for a, h in zip(self._arm_action_dims, self._hand_action_dims, strict=True)))
        # [q, qd, ee_pose(7), ee_vel(6), last_action]
        self._obs_dim = int(sum(2 * d + 13 for d in self._joint_dims) + self._action_dim)
        self.cfg.action_space = self._action_dim
        self.cfg.observation_space = self._obs_dim
        super()._configure_gym_env_spaces()
        self._actions = torch.zeros((self.num_envs, self._action_dim), device=self.device)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = torch.clamp(actions, -1.0, 1.0)

    def _apply_action(self):
        cursor = 0
        for i, robot in enumerate(self._robots):
            arm_dim = self._arm_action_dims[i]
            hand_dim = self._hand_action_dims[i]
            arm_act = self._actions[:, cursor : cursor + arm_dim]
            cursor += arm_dim
            hand_act = self._actions[:, cursor : cursor + hand_dim]
            cursor += hand_dim

            cmd = arm_act.clone()
            cmd[:, 0:3] *= self.cfg.arm_action_scale_pos
            cmd[:, 3:6] *= self.cfg.arm_action_scale_rot

            ee_pose_b = self._compute_ee_pose_b(robot=robot, ee_body_id=self._ee_body_ids[i])
            self._osc_ctrls[i].set_command(command=cmd, current_ee_pose_b=ee_pose_b)

            arm_ids = self._arm_joint_ids[i]
            jacobian_b = robot.root_physx_view.get_jacobians()[:, self._jacobi_body_ids[i], :, :][:, :, arm_ids]

            mass_matrix = robot.root_physx_view.get_generalized_mass_matrices()[:, arm_ids, :][:, :, arm_ids]
            gravity = robot.root_physx_view.get_gravity_compensation_forces()[:, arm_ids]
            ee_vel_b = self._compute_ee_velocity_b(robot=robot, ee_body_id=self._ee_body_ids[i])
            joint_pos = robot.data.joint_pos[:, arm_ids]
            joint_vel = robot.data.joint_vel[:, arm_ids]
            null_target = robot.data.default_joint_pos[:, arm_ids]

            arm_efforts = self._osc_ctrls[i].compute(
                jacobian_b=jacobian_b,
                current_ee_pose_b=ee_pose_b,
                current_ee_vel_b=ee_vel_b,
                mass_matrix=mass_matrix,
                gravity=gravity,
                current_joint_pos=joint_pos,
                current_joint_vel=joint_vel,
                nullspace_joint_pos_target=null_target,
            )
            robot.set_joint_effort_target(arm_efforts, joint_ids=arm_ids)

            hand_ids = self._hand_joint_ids[i]
            hand_target = robot.data.default_joint_pos[:, hand_ids] + self.cfg.hand_action_scale * hand_act
            robot.set_joint_position_target(hand_target, joint_ids=hand_ids)

    def _get_observations(self):
        obs_chunks = []
        for i, robot in enumerate(self._robots):
            obs_chunks.append(robot.data.joint_pos)
            obs_chunks.append(robot.data.joint_vel)
            ee_pose_b = self._compute_ee_pose_b(robot=robot, ee_body_id=self._ee_body_ids[i])
            ee_vel_b = self._compute_ee_velocity_b(robot=robot, ee_body_id=self._ee_body_ids[i])
            obs_chunks.append(ee_pose_b)
            obs_chunks.append(ee_vel_b)
        obs_chunks.append(self._actions)
        return {"policy": torch.cat(obs_chunks, dim=-1)}

    def _get_rewards(self) -> torch.Tensor:
        action_penalty = torch.sum(self._actions**2, dim=-1)
        vel_penalty = torch.zeros(self.num_envs, device=self.device)
        pose_penalty = torch.zeros(self.num_envs, device=self.device)
        for i, robot in enumerate(self._robots):
            vel_penalty += torch.sum(robot.data.joint_vel**2, dim=-1)
            ee_pose_b = self._compute_ee_pose_b(robot=robot, ee_body_id=self._ee_body_ids[i])
            pose_penalty += torch.sum(ee_pose_b[:, :3] ** 2, dim=-1)
        return (
            1.0
            - self.cfg.action_penalty_scale * action_penalty
            - self.cfg.joint_vel_penalty_scale * vel_penalty
            - self.cfg.pose_error_penalty_scale * pose_penalty
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        time_out = self.episode_length_buf >= (self.max_episode_length - 1)
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        super()._reset_idx(env_ids)
        for osc in self._osc_ctrls:
            osc.reset()
        for robot in self._robots:
            root_state = robot.data.default_root_state[env_ids_t].clone()
            root_state[:, :3] += self.scene.env_origins[env_ids_t]
            robot.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids_t)
            robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids_t)
            joint_pos = robot.data.default_joint_pos[env_ids_t].clone()
            joint_vel = robot.data.default_joint_vel[env_ids_t].clone()
            joint_pos += self.cfg.reset_joint_noise_scale * (torch.rand_like(joint_pos) - 0.5)
            robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids_t)
        self._actions[env_ids_t] = 0.0

    @staticmethod
    def _compute_ee_pose_b(robot, ee_body_id: int) -> torch.Tensor:
        ee_pos_w = robot.data.body_pos_w[:, ee_body_id]
        ee_quat_w = robot.data.body_quat_w[:, ee_body_id]
        root_pos_w = robot.data.root_pos_w
        root_quat_w = robot.data.root_quat_w
        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
        return torch.cat([ee_pos_b, ee_quat_b], dim=-1)

    @staticmethod
    def _compute_ee_velocity_b(robot, ee_body_id: int) -> torch.Tensor:
        ee_vel_w = robot.data.body_vel_w[:, ee_body_id, :]
        relative_vel_w = ee_vel_w - robot.data.root_vel_w
        ee_vel_b = torch.zeros_like(relative_vel_w)
        ee_vel_b[:, 0:3] = math_utils.quat_apply_inverse(robot.data.root_quat_w, relative_vel_w[:, 0:3])
        ee_vel_b[:, 3:6] = math_utils.quat_apply_inverse(robot.data.root_quat_w, relative_vel_w[:, 3:6])
        return ee_vel_b


def _tcp_to_wrist(ee_link: str) -> str:
    """Derive the wrist-link name from the TCP link name for AR5-style naming.

    Component names follow `..._tcp` for the tool-center-point and
    `..._link7` for the wrist. The registry surfaces the TCP as ee_link;
    the legacy env allowed picking the wrist for comparison. Works for
    AR5 components; fails loudly for components that don't follow this
    convention so the mismatch surfaces at env setup instead of deeper.
    """
    if ee_link.endswith("_tcp"):
        return ee_link[: -len("_tcp")] + "_link7"
    raise ValueError(
        f"cannot derive wrist link from ee_link={ee_link!r}: expected suffix '_tcp'. "
        f"Set cfg.ee_frame='tcp' or extend _tcp_to_wrist for this component naming."
    )
