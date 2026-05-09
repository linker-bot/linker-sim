"""Isaac `SimBackend` implementation.

Composes `isaaclab.sim.SimulationContext` + `isaaclab.scene.InteractiveScene`
directly — does not subclass `ManagerBasedRLEnv` or `DirectRLEnv` (D9).
`BaseEnv` drives this backend; controllers and tasks talk through
`self.robots[name]` / `self.rigid_bodies[name]`.

Caller responsibility: AppLauncher must have been constructed and its
`SimulationApp` must be live before this module is imported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObject, RigidObjectCfg
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
class RigidBodySpec:
    """Lightweight cfg for a task-object rigid body.

    Supports a minimal subset (box shape, mass, initial pose). Enough
    for the pick-place task in PR #2b; extend as needed.
    """

    shape: str = "box"                            # only "box" supported today
    size: tuple[float, float, float] = (0.04, 0.04, 0.04)
    mass: float = 0.05
    init_pos: tuple[float, float, float] = (0.4, 0.0, 0.05)
    init_quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    color: tuple[float, float, float] = (0.8, 0.3, 0.2)


@dataclass
class IsaacBackendCfg:
    """Per-backend settings.

    `workstations` maps a scene-level robot name (the key used in
    `backend.robots[...]`) to a workstation name registered under
    `assets/workstations/<name>/`. For now single-robot scenes are the
    only tested path; multi-robot support lands with the bimanual
    recipe in PR #3.

    `rigid_bodies` pre-declares task objects. They get spawned at scene
    construction and are reachable via `backend.rigid_bodies[name]`.
    """

    workstations: dict[str, str] = field(default_factory=lambda: {"robot": "ar5_l6_bench"})
    rigid_bodies: dict[str, RigidBodySpec] = field(default_factory=dict)
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

    Restricted to a single robot for PR #2a/b (bimanual is PR #3).
    `rigid_bodies` are attached dynamically in `__post_init__` — they
    need per-instance names but the configclass dataclass machinery
    only supports statically-declared fields, so we punt and stash
    them on a private attribute the backend reads after scene
    construction.
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
        rigid_bodies=dict(cfg.rigid_bodies),
    )


@configclass
class _SingleRobotSceneCfg(InteractiveSceneCfg):
    """Single-robot scene cfg. Consumed fields are nulled in
    `__post_init__` so `InteractiveScene`'s attribute iteration skips
    them (only asset cfgs are valid there)."""

    robot_name: str | None = "robot"
    workstation_name: str | None = "ar5_l6_bench"
    ground: bool | None = True
    dome_light: bool | None = True
    rigid_bodies: dict | None = None  # name -> RigidBodySpec

    # Concrete assets are materialized in __post_init__.
    robot: Any = None
    ground_plane: Any = None
    light: Any = None
    # Rigid-body attr names follow the pattern `obj_<name>`; created
    # dynamically in __post_init__.

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

        rigid_specs: dict = self.rigid_bodies or {}
        self._rigid_attr_names: dict[str, str] = {}
        for name, spec in rigid_specs.items():
            attr = f"obj_{name}"
            setattr(self, attr, _rigid_body_cfg(name, spec))
            self._rigid_attr_names[name] = attr

        # Null out non-asset fields so the scene iterator skips them.
        self.robot_name = None
        self.workstation_name = None
        self.ground = None
        self.dome_light = None
        self.rigid_bodies = None
        self._robot_key = robot_name  # stored for the backend to read


def _pascal(name: str) -> str:
    return "".join(p.capitalize() for p in name.replace("-", "_").split("_"))


def _rigid_body_cfg(name: str, spec: RigidBodySpec) -> RigidObjectCfg:
    """Build an `isaaclab.assets.RigidObjectCfg` from a `RigidBodySpec`."""
    if spec.shape != "box":
        raise NotImplementedError(f"rigid body shape {spec.shape!r} not yet supported")
    prim_path = "{ENV_REGEX_NS}/" + _pascal(name)
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.CuboidCfg(
            size=tuple(spec.size),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=float(spec.mass)),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(spec.color)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=tuple(spec.init_pos),
            rot=tuple(spec.init_quat_wxyz),
        ),
    )


# ---------------------------------------------------------------------------- #
# Rigid-body adapter
# ---------------------------------------------------------------------------- #


class IsaacRigidBody:
    """Adapter over `isaaclab.assets.RigidObject`."""

    def __init__(self, name: str, obj: RigidObject):
        self.name = name
        self._obj = obj
        self.num_envs = obj.num_instances
        self.device = torch.device(obj.device)

    @property
    def root_pos_w(self) -> torch.Tensor:
        return self._obj.data.root_pos_w

    @property
    def root_quat_w(self) -> torch.Tensor:
        return self._obj.data.root_quat_w

    @property
    def root_lin_vel_w(self) -> torch.Tensor:
        return self._obj.data.root_lin_vel_w

    def write_root_pose(self, pose: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        self._obj.write_root_pose_to_sim(pose, env_ids=env_ids)

    def write_root_velocity(self, velocity: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        self._obj.write_root_velocity_to_sim(velocity, env_ids=env_ids)


# ---------------------------------------------------------------------------- #
# Backend
# ---------------------------------------------------------------------------- #


class IsaacSimBackend(SimBackend):
    """Concrete Isaac backend. See module docstring."""

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

        # Read our private bookkeeping off the scene cfg BEFORE passing it
        # to InteractiveScene. IsaacLab iterates `cfg.__dict__` for asset
        # attributes and raises on any non-base-class, non-None value —
        # including our `_rigid_attr_names` dict and `_robot_key` str.
        # Null them on the cfg so the iteration skips them, but save the
        # values locally because we still need them to wire up the
        # backend's robots / rigid_bodies dicts after construction.
        scene_key = getattr(self._scene_cfg, "_robot_key", None) or "robot"
        rigid_attr_map = dict(getattr(self._scene_cfg, "_rigid_attr_names", None) or {})
        self._scene_cfg._robot_key = None
        self._scene_cfg._rigid_attr_names = None

        self._scene = InteractiveScene(self._scene_cfg)
        self._sim.reset()

        self.num_envs = int(self._scene.num_envs)
        self.device = torch.device(str(self._sim.device))
        self.dt = float(cfg.dt)
        self.env_origins = self._scene.env_origins

        self.robots: dict[str, IsaacRobot] = {}
        for role_name, ws_name in cfg.workstations.items():
            handle = load_workstation(ws_name)
            articulation = self._scene[scene_key]
            self.robots[role_name] = IsaacRobot(articulation, handle)

        self.rigid_bodies: dict[str, IsaacRigidBody] = {}
        for name, attr in rigid_attr_map.items():
            self.rigid_bodies[name] = IsaacRigidBody(name, self._scene[attr])

    # -- SimBackend interface -------------------------------------------- #

    def step(self) -> None:
        self._sim.step()
        self._scene.update(self.dt)

    def write_data(self) -> None:
        self._scene.write_data_to_sim()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        all_env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        ids = all_env_ids if env_ids is None else env_ids.to(self.device)
        if ids.numel() == 0:
            return

        for robot in self.robots.values():
            art = robot._art
            root_state = art.data.default_root_state[ids].clone()
            root_state[:, :3] += self.env_origins[ids]
            robot.write_root_state(root_state[:, :7], root_state[:, 7:], env_ids=ids)
            joint_pos = art.data.default_joint_pos[ids].clone()
            joint_vel = art.data.default_joint_vel[ids].clone()
            robot.write_joint_state(joint_pos, joint_vel, env_ids=ids)

        for body in self.rigid_bodies.values():
            obj = body._obj
            default_state = obj.data.default_root_state[ids].clone()
            default_state[:, :3] += self.env_origins[ids]
            body.write_root_pose(default_state[:, :7], env_ids=ids)
            body.write_root_velocity(default_state[:, 7:], env_ids=ids)

        self._scene.reset(ids.tolist())

    def close(self) -> None:
        pass

    # -- convenience accessors ------------------------------------------- #

    @property
    def sim(self) -> sim_utils.SimulationContext:
        return self._sim

    @property
    def scene(self) -> InteractiveScene:
        return self._scene
