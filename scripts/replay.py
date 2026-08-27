"""Replay external real-robot data through the sim.

A `ReplaySource` (see `sim/io/replay/sources.py`) supplies per-frame
joint targets keyed by composer role. This entrypoint wires it to the
MuJoCo backend and a robot, then runs `sim.runtime.replay`
— bypassing controllers, tasks, and `BaseEnv` entirely.

Usage:

    # Replay the data_collection recording on the a7_lite_l6_dc workstation
    # in the Mujoco viewer at 30 Hz wall-clock:
    python scripts/replay.py robot=a7_lite_l6_dc source=data_collection

    # Headless (no viewer):
    python scripts/replay.py robot=a7_lite_l6_dc source=data_collection \
        headless=true realtime=false max_frames=200

Config docs: `sim/configs/replay.yaml`. New recordings just need a
`sim/configs/source/<name>.yaml` describing the column layout.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _src in ("packages/linker-sim/src", "packages/linker-robot-assets/src"):
    _abs = str(REPO_ROOT / _src)
    if _abs not in sys.path:
        sys.path.insert(0, _abs)

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

OmegaConf.register_new_resolver("div", lambda a, b: a / b, replace=True)


@hydra.main(config_path="pkg://linker_sim.configs", config_name="replay", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print("[replay] resolved cfg:\n" + OmegaConf.to_yaml(cfg), flush=True)

    if cfg.backend.name != "mujoco":
        raise SystemExit(
            f"error: unknown backend {cfg.backend.name!r} (only 'mujoco' is supported)."
        )
    _replay_mujoco(cfg)


def _replay_mujoco(cfg: DictConfig) -> None:
    from linker_sim.backends.mujoco.backend import MujocoBackendCfg, MujocoSimBackend
    from linker_sim.runtime.replay import run_replay

    source = instantiate(cfg.source)
    backend = MujocoSimBackend(MujocoBackendCfg(
        workstations={cfg.robot.role_name: cfg.robot.workstation_name},
        num_envs=int(cfg.num_envs),
        dt=float(cfg.backend.dt),
        device="cpu",
    ))
    robot = backend.robots[cfg.robot.role_name]

    if cfg.headless:
        run_replay(backend, robot, source,
                   realtime=bool(cfg.realtime),
                   max_frames=cfg.max_frames)
        return

    import mujoco.viewer as mjv

    stop_flag = [False]
    restart_flag = [False]

    def on_key(keycode: int) -> None:
        if keycode in (ord("Q"), ord("q")):
            stop_flag[0] = True
        elif keycode in (ord("R"), ord("r")):
            restart_flag[0] = True

    with mjv.launch_passive(
        backend._model,
        backend._data,
        key_callback=on_key,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        _configure_mujoco_replay_camera(viewer, backend._model)
        print("[replay] hotkeys: 'R' restart, 'Q' quit")
        run_replay(backend, robot, source,
                   viewer=viewer,
                   realtime=bool(cfg.realtime),
                   max_frames=cfg.max_frames,
                   stop_flag=stop_flag,
                   loop=True,
                   restart_flag=restart_flag)


def _configure_mujoco_replay_camera(viewer, model) -> None:
    """Use a wide fixed default view so the replay robot is fully visible."""

    stat = getattr(model, "stat", None)
    center = getattr(stat, "center", [0.0, 0.0, 0.0])
    extent = float(getattr(stat, "extent", 1.0) or 1.0)
    lookat_z_offset = float(os.environ.get("MUJOCO_REPLAY_CAMERA_LOOKAT_Z_OFFSET", "0.10"))

    distance_env = os.environ.get("MUJOCO_REPLAY_CAMERA_DISTANCE", "").strip()
    if distance_env:
        distance = float(distance_env)
    else:
        distance_scale = float(os.environ.get("MUJOCO_REPLAY_CAMERA_DISTANCE_SCALE", "1.0667"))
        distance = max(1.8333, extent * distance_scale)

    with viewer.lock():
        viewer.cam.lookat[0] = float(center[0])
        viewer.cam.lookat[1] = float(center[1])
        viewer.cam.lookat[2] = float(center[2]) + extent * lookat_z_offset
        viewer.cam.distance = distance
        viewer.cam.azimuth = float(os.environ.get("MUJOCO_REPLAY_CAMERA_AZIMUTH", "180"))
        viewer.cam.elevation = float(os.environ.get("MUJOCO_REPLAY_CAMERA_ELEVATION", "-15"))


if __name__ == "__main__":
    main()
