"""Replay external real-robot data through the sim.

A `ReplaySource` (see `sim/io/replay/sources.py`) supplies per-frame
joint targets keyed by composer role. This entrypoint wires it to a
backend (mujoco or isaac) and a robot, then runs `sim.runtime.replay`
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

import argparse
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

    if cfg.backend.name == "mujoco":
        _replay_mujoco(cfg)
    elif cfg.backend.name == "viser":
        _replay_viser(cfg)
    else:
        _replay_isaac(cfg)


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


def _replay_viser(cfg: DictConfig) -> None:
    from linker_sim.backends.viser.backend import ViserBackendCfg, ViserSimBackend
    from linker_sim.runtime.replay import run_replay

    source = instantiate(cfg.source)
    backend = ViserSimBackend(ViserBackendCfg(
        workstations={cfg.robot.role_name: cfg.robot.workstation_name},
        num_envs=int(cfg.num_envs),
        dt=float(cfg.backend.dt),
        device="cpu",
        host=str(cfg.backend.host),
        port=int(cfg.backend.port),
        headless=bool(cfg.headless),
    ))
    try:
        robot = backend.robots[cfg.robot.role_name]
        run_replay(
            backend, robot, source,
            realtime=bool(cfg.realtime),
            max_frames=cfg.max_frames,
            loop=not bool(cfg.headless),
        )
    finally:
        backend.close()


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


def _replay_isaac(cfg: DictConfig) -> None:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(add_help=False)
    AppLauncher.add_app_launcher_args(parser)
    launch_args = parser.parse_args([])
    launch_args.headless = bool(cfg.headless)
    launch_args.device = str(cfg.device)
    app_launcher = AppLauncher(launch_args)
    simulation_app = app_launcher.app

    import traceback
    try:
        import carb.input
        import omni.appwindow

        from linker_sim.backends.isaac.backend import IsaacBackendCfg, IsaacSimBackend
        from linker_sim.runtime.replay import run_replay

        rigid_bodies = {}
        if getattr(cfg.robot, "rigid_bodies", None):
            for name, spec in cfg.robot.rigid_bodies.items():
                rigid_bodies[name] = instantiate(spec)

        source = instantiate(cfg.source)
        backend = IsaacSimBackend(IsaacBackendCfg(
            workstations={cfg.robot.role_name: cfg.robot.workstation_name},
            rigid_bodies=rigid_bodies,
            num_envs=int(cfg.num_envs),
            env_spacing=float(cfg.get("env_spacing", 2.5)),
            dt=float(cfg.backend.dt),
            render_interval=int(cfg.backend.render_interval),
            device=str(cfg.device),
            ground=bool(cfg.backend.ground),
            dome_light=bool(cfg.backend.dome_light),
        ))
        robot = backend.robots[cfg.robot.role_name]

        stop_flag = [False]
        restart_flag = [False]

        app_window = omni.appwindow.get_default_app_window()
        input_iface = carb.input.acquire_input_interface()
        keyboard = app_window.get_keyboard()

        def _on_key(event, *args):
            if event.type == carb.input.KeyboardEventType.KEY_PRESS:
                if event.input == carb.input.KeyboardInput.R:
                    restart_flag[0] = True
                elif event.input == carb.input.KeyboardInput.Q:
                    stop_flag[0] = True
            return True

        kb_sub = input_iface.subscribe_to_keyboard_events(keyboard, _on_key)
        print("[replay] hotkeys: 'R' restart, 'Q' quit")

        run_replay(backend, robot, source,
                   realtime=bool(cfg.realtime),
                   max_frames=cfg.max_frames,
                   stop_flag=stop_flag,
                   loop=True,
                   restart_flag=restart_flag)

        input_iface.unsubscribe_to_keyboard_events(keyboard, kb_sub)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
