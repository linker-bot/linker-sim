"""Replay a recorded episode through a live env.

Usage:

    # Re-drive the actions through a reach env:
    python scripts/replay.py episode=./outputs/.../episodes/episode_000000.jsonl

    # Inject recorded joint state:
    python scripts/replay.py episode=./...jsonl mode=state_inject
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

from isaaclab.app import AppLauncher

OmegaConf.register_new_resolver("div", lambda a, b: a / b, replace=True)


@hydra.main(config_path=str(REPO_ROOT / "sim" / "configs"), config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    if "episode" not in cfg:
        raise SystemExit("error: specify +episode=<path> on the CLI")

    parser = argparse.ArgumentParser(add_help=False)
    AppLauncher.add_app_launcher_args(parser)
    launch_args = parser.parse_args([])
    launch_args.headless = bool(cfg.get("headless", False))
    launch_args.device = str(cfg.device)
    app_launcher = AppLauncher(launch_args)
    simulation_app = app_launcher.app

    import traceback
    try:
        _replay(cfg)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()


def _replay(cfg: DictConfig) -> None:
    from sim.backends.isaac.backend import IsaacBackendCfg, IsaacSimBackend
    from sim.envs.base import BaseEnv, BaseEnvCfg
    from sim.io.replayer import ReplayEpisode, Replayer

    mode = cfg.get("mode", "action_replay")

    rigid_bodies = {}
    if getattr(cfg.robot, "rigid_bodies", None):
        for name, spec in cfg.robot.rigid_bodies.items():
            rigid_bodies[name] = instantiate(spec)

    backend = IsaacSimBackend(IsaacBackendCfg(
        workstations={cfg.robot.role_name: cfg.robot.workstation_name},
        rigid_bodies=rigid_bodies,
        num_envs=int(cfg.num_envs),
        env_spacing=float(cfg.env_spacing),
        dt=float(cfg.backend.dt),
        render_interval=int(cfg.backend.render_interval),
        device=str(cfg.device),
        ground=bool(cfg.backend.ground),
        dome_light=bool(cfg.backend.dome_light),
    ))
    controllers = [instantiate(entry) for entry in cfg.controller.entries]
    task = instantiate(cfg.task, backend=backend)
    env = BaseEnv(backend, controllers, task, BaseEnvCfg(
        robot_name=cfg.robot.role_name,
        decimation=int(cfg.decimation),
        episode_length_s=float(cfg.episode_length_s),
        reset_joint_noise_scale=0.0,  # deterministic replay
    ))

    episode = ReplayEpisode.load_jsonl(Path(cfg.episode))
    replayer = Replayer(env, mode=mode)
    print(f"[replay] mode={mode} episode_id={episode.episode_id} frames={len(episode.frames)}")
    for _ in replayer.replay(episode):
        pass


if __name__ == "__main__":
    main()
