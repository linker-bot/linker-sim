"""OSC RL env over the PR #2 runtime backbone.

PR #2a rewires this away from IsaacLab's `DirectRLEnv`. The class is
now a thin wrapper around
`sim.envs.base.BaseEnv + sim.controllers.{OscController, JointPDController}
+ sim.envs.base.LegacyOscTask`. The public surface (class name, cfg
fields used by `gain_tuner_osc.py`) is preserved only where callers
still reach in; the gym-style API (`reset`, `step`) is new.

Caller responsibility: a `SimulationApp` must be live before this
module is imported. Every Isaac-dependent script does this at the top
via `AppLauncher`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch

from sim.backends.isaac.backend import IsaacBackendCfg, IsaacSimBackend
from sim.controllers.joint_pd import JointPDController, JointPDControllerCfg
from sim.controllers.osc import OscController, OscControllerCfg
from sim.envs.base import BaseEnv, BaseEnvCfg, LegacyOscTask, LegacyOscTaskCfg


@dataclass
class TestOscRLEnvCfg:
    """OSC-first env config using EEF pose deltas + hand action channel.

    Minimal set of knobs — most controller-level settings live on
    `OscControllerCfg` / `JointPDControllerCfg` now. Kept here for
    backward compatibility with the legacy CLI wrappers.
    """

    num_envs: int = 64
    env_spacing: float = 2.5
    workstation_name: str = "ar5_l6_bench"
    ee_frame: Literal["tcp", "wrist"] = "tcp"

    decimation: int = 4
    episode_length_s: float = 8.0
    dt: float = 1.0 / 120.0
    device: str = "cuda:0"

    arm_action_scale_pos: float = 0.05
    arm_action_scale_rot: float = 0.25
    hand_action_scale: float = 0.2
    reset_joint_noise_scale: float = 0.02

    action_penalty_scale: float = 0.01
    joint_vel_penalty_scale: float = 0.001
    pose_error_penalty_scale: float = 1.0

    osc_stiffness: tuple[float, ...] = (150.0, 150.0, 150.0, 80.0, 80.0, 80.0)
    osc_damping_ratio: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    osc_nullspace_stiffness: float = 10.0

    # Actuator-gain override applied to the arm role on OSC attach
    # (lower PD so effort commands dominate).
    osc_actuator_stiffness: float = 150.0
    osc_actuator_damping: float = 8.0


class TestOscRLEnv:
    """OSC control env. Actions are EEF deltas + hand position deltas.

    This is a gym-style wrapper now; it does not subclass `DirectRLEnv`.
    Callers that held a reference to the legacy class should migrate to
    `BaseEnv` directly; this shim exists so the class name keeps
    resolving during the PR #2a → #2b transition.
    """

    def __init__(self, cfg: TestOscRLEnvCfg | None = None, **_ignored):
        self.cfg = cfg or TestOscRLEnvCfg()

        backend_cfg = IsaacBackendCfg(
            workstations={"robot": self.cfg.workstation_name},
            num_envs=self.cfg.num_envs,
            env_spacing=self.cfg.env_spacing,
            dt=self.cfg.dt,
            render_interval=self.cfg.decimation,
            device=self.cfg.device,
        )
        self.backend = IsaacSimBackend(backend_cfg)

        ee_frame = "arm:tool0" if self.cfg.ee_frame == "tcp" else None
        if self.cfg.ee_frame == "wrist":
            # No handle.frames entry for the wrist; fall back to the
            # legacy naming convention (see PR1_PROGRESS.md sharp edges).
            handle = self.backend.robots["robot"].handle
            ee = handle.ee_link
            if not ee.endswith("_tcp"):
                raise ValueError(
                    f"cfg.ee_frame='wrist' requires ee_link ending in '_tcp' "
                    f"(got {ee!r}). Extend handle.frames to expose a named "
                    f"wrist frame."
                )
            ee_frame = ee[: -len("_tcp")] + "_link7"

        osc = OscController(
            OscControllerCfg(
                role="arm",
                frame=ee_frame,
                action_scale_pos=self.cfg.arm_action_scale_pos,
                action_scale_rot=self.cfg.arm_action_scale_rot,
                stiffness=self.cfg.osc_stiffness,
                damping_ratio=self.cfg.osc_damping_ratio,
                nullspace_stiffness=self.cfg.osc_nullspace_stiffness,
                actuator_stiffness=self.cfg.osc_actuator_stiffness,
                actuator_damping=self.cfg.osc_actuator_damping,
            )
        )
        hand = JointPDController(
            JointPDControllerCfg(role="hand", action_scale=self.cfg.hand_action_scale)
        )

        # Attach order is important for command-slicing: arm command
        # comes first in the action vector (OSC 6 dims), then hand.
        task = LegacyOscTask(
            self.backend.robots["robot"],
            LegacyOscTaskCfg(
                ee_frame=ee_frame,
                action_penalty_scale=self.cfg.action_penalty_scale,
                joint_vel_penalty_scale=self.cfg.joint_vel_penalty_scale,
                pose_error_penalty_scale=self.cfg.pose_error_penalty_scale,
                action_dim=osc.command_dim + len(self.backend.robots["robot"].handle.joints["hand"]),
            ),
        )
        self.env = BaseEnv(
            self.backend,
            controllers=[osc, hand],
            task=task,
            cfg=BaseEnvCfg(
                robot_name="robot",
                decimation=self.cfg.decimation,
                episode_length_s=self.cfg.episode_length_s,
                reset_joint_noise_scale=self.cfg.reset_joint_noise_scale,
            ),
        )

    # -- gym API --------------------------------------------------------- #

    @property
    def action_dim(self) -> int:
        return self.env.action_dim

    @property
    def observation_dim(self) -> int:
        return self.env.observation_dim

    @property
    def num_envs(self) -> int:
        return self.env.num_envs

    @property
    def device(self) -> torch.device:
        return self.env.device

    def reset(
        self,
        seed: int | None = None,
        env_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        return self.env.reset(seed=seed, env_ids=env_ids)

    def step(self, action: torch.Tensor):
        return self.env.step(action)

    def close(self) -> None:
        self.backend.close()
