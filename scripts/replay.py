"""Replay external real-robot data through the sim.

A `ReplaySource` (see `sim/io/replay/sources.py`) supplies per-frame
joint targets keyed by composer role. This entrypoint wires it to a
backend (mujoco or isaac) and a robot, then runs `sim.runtime.replay`
— bypassing controllers, tasks, and `BaseEnv` entirely.

Usage:

    # Replay the data_collection recording on the a7_lite_dc workstation
    # in the Mujoco viewer at 30 Hz wall-clock:
    python scripts/replay.py robot=a7_lite_dc source=data_collection

    # Headless (no viewer):
    python scripts/replay.py robot=a7_lite_dc source=data_collection \
        headless=true realtime=false max_frames=200

Config docs: `sim/configs/replay.yaml`. New recordings just need a
`sim/configs/source/<name>.yaml` describing the column layout.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

OmegaConf.register_new_resolver("div", lambda a, b: a / b, replace=True)


@hydra.main(config_path=str(REPO_ROOT / "sim" / "configs"), config_name="replay", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print("[replay] resolved cfg:\n" + OmegaConf.to_yaml(cfg), flush=True)

    if cfg.backend.name == "mujoco":
        _replay_mujoco(cfg)
    else:
        _replay_isaac(cfg)


def _replay_mujoco(cfg: DictConfig) -> None:
    from sim.backends.mujoco.backend import MujocoBackendCfg, MujocoSimBackend
    from sim.runtime.replay import run_replay

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

    def on_key(keycode: int) -> None:
        if keycode in (ord("Q"), ord("q")):
            stop_flag[0] = True

    with mjv.launch_passive(backend._model, backend._data, key_callback=on_key) as viewer:
        print("[replay] hotkey: press 'Q' in the viewport to stop")
        run_replay(backend, robot, source,
                   viewer=viewer,
                   realtime=bool(cfg.realtime),
                   max_frames=cfg.max_frames,
                   stop_flag=stop_flag)


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
        from sim.backends.isaac.backend import IsaacBackendCfg, IsaacSimBackend
        from sim.runtime.replay import run_replay

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

        run_replay(backend, robot, source,
                   realtime=bool(cfg.realtime),
                   max_frames=cfg.max_frames)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
