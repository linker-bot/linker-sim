"""Isaac `SimBackend` implementation.

Composes `isaaclab.sim.SimulationContext` + `isaaclab.scene.InteractiveScene`
directly — does not subclass `ManagerBasedRLEnv` or `DirectRLEnv` (D9).
`BaseEnv` drives this backend; controllers and tasks talk through
`self.robots[name]` which are `IsaacRobot`s.

The scene builder (`build_scene_cfg`) lives here too — scene assembly
is a backend concern, not an env concern, and moving it out of
`sim/envs/test_osc/scene_cfg.py` lets every backend own its native
scene primitives.

Caller responsibility: AppLauncher must have been constructed and its
`SimulationApp` must be live before this module is imported. Every
Isaac-dependent script does this at the top; see
`sim/envs/test_osc/spawn_osc_scene.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

from sim.backends.base import SimBackend
from sim.backends.isaac.loaders import to_articulation_cfg
from sim.backends.isaac.robot import IsaacRobot
from sim.registry import load as load_workstation


# ---------------------------------------------------------------------------- #
# Config
# ---------------------------------------------------------------------------- #


@dataclass
class IsaacBackendCfg:
    """Per-backend settings.

    `workstations` maps a scene-level robot name (the key used in
    `backend.robots[...]`) to a workstation name registered under
    `assets/workstations/<name>/`. For now single-robot scenes are the
    only tested path; multi-robot support lands with the bimanual
    recipe in PR #3.
    """

    workstations: dict[str, str] = field(default_factory=lambda: {"robot": "ar5_l6_bench"})
    num_envs: int = 1
    env_spacing: float = 2.5
    dt: float = 1.0 / 120.0
    render_interval: int = 4
    device: str = "cuda:0"
    ground: bool = True
    dome_light: bool = True


# ---------------------------------------------------------------------------- #
# Scene cfg builder
# ---------------------------------------------------------------------------- #


def build_scene_cfg(cfg: IsaacBackendCfg) -> InteractiveSceneCfg:
    """Assemble an `InteractiveSceneCfg` for `cfg.workstations`.

    Restricted to a single robot for PR #2a. Multi-robot scenes will
    require either a dynamically-built configclass (messy) or a shift
    to multiple articulations per env (bimanual recipe path). For now
    we enforce one entry and build the tidy single-robot class below.
    """
    if len(cfg.workstations) != 1:
        raise NotImplementedError(
            "Multi-robot scenes require a bimanual workstation recipe "
            "(deferred to PR #3). Use a single composed workstation for now."
        )
    (robot_name, ws_name), = cfg.workstations.items()
    return _SingleRobotSceneCfg(
        num_envs=cfg.num_envs,
        env_spacing=cfg.env_spacing,
        robot_name=robot_name,
        workstation_name=ws_name,
        ground=cfg.ground,
        dome_light=cfg.dome_light,
    )


@configclass
class _SingleRobotSceneCfg(InteractiveSceneCfg):
    """Single-robot scene. `robot_name` / `workstation_name` are consumed
    in `__post_init__` and then set to `None` so `InteractiveScene`'s
    attribute iteration skips them (only asset cfgs are valid there)."""

    robot_name: str | None = "robot"
    workstation_name: str | None = "ar5_l6_bench"
    ground: bool | None = True
    dome_light: bool | None = True

    # Concrete assets are materialized in __post_init__.
    robot: Any = None
    ground_plane: Any = None
    light: Any = None

    def __post_init__(self):
        super().__post_init__()
        ws_name = self.workstation_name or "ar5_l6_bench"
        robot_name = self.robot_name or "robot"
        handle = load_workstation(ws_name)
        self.robot = to_articulation_cfg(
            handle, prim_path="{ENV_REGEX_NS}/" + _pascal(robot_name)
        )
        if self.ground:
            self.ground_plane = AssetBaseCfg(
                prim_path="/World/defaultGroundPlane",
                spawn=sim_utils.GroundPlaneCfg(),
            )
        if self.dome_light:
            self.light = AssetBaseCfg(
                prim_path="/World/Light",
                spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
            )
        # Null out everything that isn't an asset cfg so
        # InteractiveScene._add_entities_from_cfg skips it.
        self.robot_name = None
        self.workstation_name = None
        self.ground = None
        self.dome_light = None
        # The scene pulls entities by attribute name; expose the robot
        # under the requested key. configclass doesn't let us set
        # arbitrary names at class-def time, so we punt by always using
        # the canonical attribute `robot` and having the backend look it
        # up under that name (see IsaacSimBackend._scene_robot_key).
        self._robot_key = robot_name  # stored for the backend to read


def _pascal(name: str) -> str:
    return "".join(p.capitalize() for p in name.replace("-", "_").split("_"))


# ---------------------------------------------------------------------------- #
# Backend
# ---------------------------------------------------------------------------- #


class IsaacSimBackend(SimBackend):
    """Concrete Isaac backend.

    Owns the `SimulationContext` + `InteractiveScene`. Exposes
    `self.robots[name] -> IsaacRobot`. Driver (`BaseEnv` or a CLI
    script) calls `step()` / `reset()` / `write_data()` in a loop.

    Construction order mirrors the manual script:
        SimulationContext(cfg.sim)   # physics dt, device, render
        InteractiveScene(scene_cfg)  # spawns prims
        sim.reset()                  # starts physx + populates handles
    """

    def __init__(
        self,
        cfg: IsaacBackendCfg,
        *,
        scene_cfg: InteractiveSceneCfg | None = None,
    ):
        self.cfg = cfg
        sim_cfg = sim_utils.SimulationCfg(
            dt=cfg.dt,
            render_interval=cfg.render_interval,
            device=cfg.device,
        )
        self._sim = sim_utils.SimulationContext(sim_cfg)
        self._sim.set_camera_view(eye=[1.8, -1.4, 1.2], target=[0.4, 0.0, 0.4])

        self._scene_cfg = scene_cfg if scene_cfg is not None else build_scene_cfg(cfg)
        self._scene = InteractiveScene(self._scene_cfg)
        self._sim.reset()

        self.num_envs = int(self._scene.num_envs)
        self.device = torch.device(str(self._sim.device))
        self.dt = float(cfg.dt)
        self.env_origins = self._scene.env_origins

        # Build Robot adapters. The scene indexes articulations by their
        # attribute name on the scene cfg; `_SingleRobotSceneCfg` parks
        # the articulation under `robot` and stashes the user-facing
        # role name in `_robot_key`.
        scene_key = getattr(self._scene_cfg, "_robot_key", None) or "robot"
        self.robots: dict[str, IsaacRobot] = {}
        for role_name, ws_name in cfg.workstations.items():
            handle = load_workstation(ws_name)
            # Only `scene_key` lives in the scene for a single-robot cfg;
            # future multi-robot impls will build a per-role lookup here.
            articulation = self._scene[scene_key]
            self.robots[role_name] = IsaacRobot(articulation, handle)

    # -- SimBackend interface -------------------------------------------- #

    def step(self) -> None:
        self._sim.step()
        self._scene.update(self.dt)

    def write_data(self) -> None:
        self._scene.write_data_to_sim()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset `env_ids` (or all envs) to the default articulation
        state. Controllers should be reset separately by the env."""
        all_env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        ids = all_env_ids if env_ids is None else env_ids.to(self.device)
        if ids.numel() == 0:
            return

        for robot in self.robots.values():
            art = robot._art  # fine inside the same subsystem
            root_state = art.data.default_root_state[ids].clone()
            root_state[:, :3] += self.env_origins[ids]
            robot.write_root_state(root_state[:, :7], root_state[:, 7:], env_ids=ids)
            joint_pos = art.data.default_joint_pos[ids].clone()
            joint_vel = art.data.default_joint_vel[ids].clone()
            robot.write_joint_state(joint_pos, joint_vel, env_ids=ids)

        # IsaacLab caches some per-scene buffers; the scene's own reset
        # handles those. Pass a python list per its signature.
        self._scene.reset(ids.tolist())

    def close(self) -> None:
        # SimulationContext has no explicit close (the SimulationApp
        # owns the lifecycle); this is a no-op here so callers can treat
        # the backend as a context-managed resource uniformly.
        pass

    # -- convenience accessors ------------------------------------------- #

    @property
    def sim(self) -> sim_utils.SimulationContext:
        return self._sim

    @property
    def scene(self) -> InteractiveScene:
        return self._scene
