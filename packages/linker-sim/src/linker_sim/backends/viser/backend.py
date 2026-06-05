"""Viser `SimBackend` implementation (replay-only).

Loads a composed `workstation.urdf` via `linker_sim.registry`, opens a
Viser scene at `cfg.host:cfg.port`, and animates the URDF as
`run_replay()` writes joint targets each frame. There is no physics
loop — `step()` is a no-op; `write_data()` pushes the current joint
buffer into the Viser scene.

Headless mode (`cfg.headless=True`) skips ViserServer construction
entirely so unit tests don't bind a port.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from linker_sim.backends.viser.robot import ViserRobot
from linker_sim.registry import load as load_workstation

try:
    import viser
    import viser.extras
    import yourdfpy
except ImportError:
    viser = None  # type: ignore[assignment]
    yourdfpy = None  # type: ignore[assignment]


@dataclass
class ViserBackendCfg:
    workstations: dict[str, str] = field(default_factory=lambda: {"robot": "ar5_o6_bench_bimanual"})
    num_envs: int = 1
    dt: float = 1.0 / 30.0
    device: str = "cpu"
    host: str = "127.0.0.1"
    port: int = 8080
    headless: bool = False


class ViserSimBackend:
    """Replay-only Viser backend. See module docstring."""

    def __init__(self, cfg: ViserBackendCfg):
        if viser is None or yourdfpy is None:
            raise ImportError(
                "viser and/or yourdfpy are not installed. Install with "
                "`pip install -e packages/linker-robot-assets -e 'packages/linker-sim[viser]'`."
            )
        if cfg.num_envs != 1:
            raise NotImplementedError(
                f"Viser backend supports num_envs=1 only (got {cfg.num_envs})."
            )
        if str(cfg.device) != "cpu":
            raise ValueError(
                f"Viser backend requires device='cpu' (got {cfg.device!r})"
            )
        if len(cfg.workstations) != 1:
            raise NotImplementedError(
                "Multi-articulation scenes are not supported. Use one composed "
                "workstation per scene."
            )

        self.cfg = cfg
        (robot_name, ws_name), = cfg.workstations.items()

        handle = load_workstation(ws_name)
        self._urdf = yourdfpy.URDF.load(str(handle.urdf_path))

        if cfg.headless:
            self._server = None
            self._viser_urdf = None
        else:
            self._server = viser.ViserServer(host=cfg.host, port=cfg.port)
            self._viser_urdf = viser.extras.ViserUrdf(self._server, self._urdf)
            print(f"[viser] scene at http://{cfg.host}:{cfg.port}", flush=True)

        self.num_envs = 1
        self.device = torch.device("cpu")
        self.dt = float(cfg.dt)
        self.env_origins = torch.zeros(1, 3)

        robot = ViserRobot(handle=handle, urdf=self._urdf)
        self.robots: dict[str, ViserRobot] = {robot_name: robot}
        self.rigid_bodies: dict = {}
        self._robot = robot  # convenience back-ref for write_data

    def step(self) -> None:
        # No physics. The replay loop's own realtime pacing handles wall-clock.
        pass

    def write_data(self) -> None:
        if self._viser_urdf is None:
            return
        self._viser_urdf.update_cfg(self._robot.joint_buffer)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        del env_ids
        self._robot.reset_to_default()
        self.write_data()

    def close(self) -> None:
        if self._server is not None:
            try:
                self._server.stop()
            except Exception:
                pass
            self._server = None
            self._viser_urdf = None
