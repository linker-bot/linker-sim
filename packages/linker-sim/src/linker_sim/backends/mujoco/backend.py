"""MuJoCo `SimBackend` implementation.

Loads a composed `workstation.mjcf` via `sim.registry`, runs B=1 physics
on CPU, and exposes one `MujocoRobot` per cfg.workstations entry.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from linker_sim.backends.mujoco.robot import MujocoRobot
from linker_sim.registry import load as load_workstation

try:
    import mujoco
except ImportError:
    mujoco = None  # type: ignore[assignment]


@dataclass
class MujocoBackendCfg:
    workstations: dict[str, str] = field(default_factory=lambda: {"robot": "ar5_o6_bench_bimanual"})
    num_envs: int = 1
    dt: float = 1.0 / 500.0
    device: str = "cpu"


class MujocoSimBackend:
    """Concrete MuJoCo backend. See module docstring."""

    def __init__(self, cfg: MujocoBackendCfg):
        if mujoco is None:
            raise ImportError(
                "mujoco is not installed. Install with "
                "`pip install 'linker-sim[mujoco]'`."
            )
        if cfg.num_envs != 1:
            raise NotImplementedError(
                f"MuJoCo backend supports num_envs=1 only (got {cfg.num_envs}). "
                "Parallel rollouts use one process per env (D8)."
            )
        if str(cfg.device) != "cpu":
            raise ValueError(
                f"MuJoCo backend requires device='cpu' (got {cfg.device!r})"
            )
        if len(cfg.workstations) != 1:
            raise NotImplementedError(
                "Multi-articulation scenes are not supported. Use one composed "
                "workstation per scene, e.g. workstations={'robot': 'ar5_o6_bench_bimanual'}."
            )

        self.cfg = cfg
        (robot_name, ws_name), = cfg.workstations.items()

        handle = load_workstation(ws_name)
        if handle.mjcf_path is None:
            raise FileNotFoundError(
                f"{ws_name}: workstation.mjcf missing — run "
                f"`python -m linker_robot_assets.composer.compose assets/workstations/{ws_name}`"
            )

        self._model = mujoco.MjModel.from_xml_path(str(handle.mjcf_path))
        self._model.opt.timestep = float(cfg.dt)
        self._data = mujoco.MjData(self._model)
        mujoco.mj_forward(self._model, self._data)

        self.num_envs = 1
        self.device = torch.device("cpu")
        self.dt = float(cfg.dt)
        self.env_origins = torch.zeros(1, 3)

        robot = MujocoRobot(self._model, self._data, handle)
        self.robots: dict[str, MujocoRobot] = {robot_name: robot}
        self.rigid_bodies: dict = {}

        self._default_qpos = self._model.qpos0.copy()
        self._default_qvel = np.zeros(self._model.nv, dtype=np.float64)

    def step(self) -> None:
        mujoco.mj_step(self._model, self._data)

    def write_data(self) -> None:
        pass

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        del env_ids
        self._data.qpos[:] = self._default_qpos
        self._data.qvel[:] = self._default_qvel
        self._data.ctrl[:] = 0.0
        self._data.qfrc_applied[:] = 0.0
        mujoco.mj_forward(self._model, self._data)

    def close(self) -> None:
        pass
