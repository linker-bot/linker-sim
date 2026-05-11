"""Hydra-driven entrypoint for the runtime backbone.

Usage:

    # Smoke rollout with OSC on a reach task:
    python scripts/run.py

    # Pick-and-place with JSONL recording:
    python scripts/run.py task=pick_place recorder=jsonl

    # Different workstation:
    python scripts/run.py robot=ar5_l6_bench_right

    # Joint-PD on both roles (override controller group):
    python scripts/run.py controller=joint_pd task=reach

    # Override arbitrary fields:
    python scripts/run.py num_envs=4 max_steps=500 policy=random_walk

Hydra docs: https://hydra.cc/docs/intro/
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

# Register the div resolver used in configs (rationals for dt).
OmegaConf.register_new_resolver("div", lambda a, b: a / b, replace=True)


def _launch_isaac(cfg: DictConfig):
    parser = argparse.ArgumentParser(add_help=False)
    AppLauncher.add_app_launcher_args(parser)
    launch_args = parser.parse_args([])
    launch_args.headless = bool(cfg.headless)
    launch_args.device = str(cfg.device)
    app_launcher = AppLauncher(launch_args)
    return app_launcher.app


@hydra.main(config_path=str(REPO_ROOT / "sim" / "configs"), config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print("[run] resolved cfg:\n" + OmegaConf.to_yaml(cfg), flush=True)

    if cfg.backend.name == "mujoco":
        raise SystemExit(
            "error: backend=mujoco is a stub. Blocked on PR #1b "
            "(component MJCF authoring). Use backend=isaac."
        )

    simulation_app = _launch_isaac(cfg)
    import traceback
    try:
        _run_isaac(cfg)
    except BaseException:
        # Surface the traceback BEFORE Kit's shutdown prints clobber
        # stderr — same pattern as sim/envs/test_osc/spawn_osc_scene.py
        # (see docs/PR1_PROGRESS.md for the original diagnosis).
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()


def _run_isaac(cfg: DictConfig) -> None:
    # Imports that require the SimulationApp to be live.
    import torch
    from sim.backends.isaac.backend import IsaacBackendCfg, IsaacSimBackend
    from sim.envs.base import BaseEnv, BaseEnvCfg
    from sim.io.recorder import Recorder

    # --- Backend ----------------------------------------------------------- #
    rigid_bodies = {}
    if getattr(cfg.robot, "rigid_bodies", None):
        for name, spec in cfg.robot.rigid_bodies.items():
            rigid_bodies[name] = instantiate(spec)

    backend_cfg = IsaacBackendCfg(
        workstations={cfg.robot.role_name: cfg.robot.workstation_name},
        rigid_bodies=rigid_bodies,
        num_envs=int(cfg.num_envs),
        env_spacing=float(cfg.env_spacing),
        dt=float(cfg.backend.dt),
        render_interval=int(cfg.backend.render_interval),
        device=str(cfg.device),
        ground=bool(cfg.backend.ground),
        dome_light=bool(cfg.backend.dome_light),
    )
    backend = IsaacSimBackend(backend_cfg)
    print(f"[run] backend ready: {cfg.robot.workstation_name} x{backend.num_envs}")

    # --- Controllers ------------------------------------------------------- #
    controllers = [instantiate(entry) for entry in cfg.controller.entries]

    # --- Task -------------------------------------------------------------- #
    # _recursive_=True (default) materializes the nested `cfg:` as a
    # ReachTaskCfg / PickPlaceTaskCfg instance before calling the task's
    # __init__. Without it, `cfg:` stays as a DictConfig and breaks the
    # isinstance check.
    task = instantiate(cfg.task, backend=backend)

    # --- Env --------------------------------------------------------------- #
    env_cfg = BaseEnvCfg(
        robot_name=cfg.robot.role_name,
        decimation=int(cfg.decimation),
        episode_length_s=float(cfg.episode_length_s),
        reset_joint_noise_scale=float(cfg.reset_joint_noise_scale),
    )
    env = BaseEnv(backend, controllers, task, env_cfg)
    print(f"[run] env ready: action_dim={env.action_dim} observation_dim={env.observation_dim}")

    # --- Recorder ---------------------------------------------------------- #
    # `cfg.recorder` may be absent (if the defaults list mapping is weird)
    # or None (if someone set recorder=null on the CLI). Tolerate both.
    recorder_cfg = OmegaConf.select(cfg, "recorder", default=None)
    recorder: Recorder | None = None
    if recorder_cfg is not None:
        recorder = instantiate(recorder_cfg, num_envs=backend.num_envs)

    # --- Rollout ----------------------------------------------------------- #
    obs, _ = env.reset(seed=int(cfg.seed))
    policy = _make_policy(cfg.policy, env.action_dim, backend.num_envs, backend.device)

    # Hotkey: press 'R' in the Isaac viewport to reset all envs.
    # No-op in headless mode (no keyboard subscription possible).
    reset_flag = _register_hotkey_reset(cfg.headless)

    step = 0
    # max_steps <= 0 → run until the user closes the window.
    # BaseEnv already resets done envs in-place, so the loop keeps going
    # across episode boundaries without any extra bookkeeping here.
    max_steps = int(cfg.max_steps) if int(cfg.max_steps) > 0 else None

    # Import here to avoid binding at module load.
    simulation_app = _get_simulation_app()

    while simulation_app.is_running():
        if max_steps is not None and step >= max_steps:
            break
        if reset_flag[0]:
            reset_flag[0] = False
            obs, _ = env.reset()
            print(f"[run] manual reset at step {step}")
        action = policy(step, obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if recorder is not None:
            recorder.record_step(obs, action, reward, terminated, truncated, info)
        step += 1

    if recorder is not None:
        recorder.close()
    print(f"[run] done after {step} steps.")


def _get_simulation_app():
    # Avoid re-importing AppLauncher (which re-starts the app).
    import omni.kit.app  # type: ignore
    return omni.kit.app.get_app()


def _register_hotkey_reset(headless: bool, key_name: str = "R") -> list:
    """Subscribe to viewport key presses; flip the returned flag on `key_name`.

    Returns a 1-element list used as a mutable bool (set to True on press,
    consumed by the main loop). In headless mode the subscription is
    skipped and a dead flag is returned so the main loop can check it
    unconditionally.
    """
    flag = [False]
    if headless:
        return flag

    import carb.input  # type: ignore
    import omni.appwindow  # type: ignore

    target_key = getattr(carb.input.KeyboardInput, key_name.upper())

    def on_event(event, *_):
        if (
            event.type == carb.input.KeyboardEventType.KEY_PRESS
            and event.input == target_key
        ):
            flag[0] = True
        return True

    appwindow = omni.appwindow.get_default_app_window()
    keyboard = appwindow.get_keyboard()
    input_iface = carb.input.acquire_input_interface()
    input_iface.subscribe_to_keyboard_events(keyboard, on_event)
    print(f"[run] hotkey: press '{key_name.upper()}' in the viewport to reset")
    return flag


def _make_policy(name: str, action_dim: int, num_envs: int, device):
    import torch

    if name == "zeros":
        def policy(_step, _obs):
            return torch.zeros((num_envs, action_dim), device=device)
    elif name == "random_walk":
        def policy(_step, _obs):
            return 0.1 * (2.0 * torch.rand((num_envs, action_dim), device=device) - 1.0)
    else:
        raise ValueError(f"unknown policy: {name!r} (expected 'zeros' or 'random_walk')")
    return policy


if __name__ == "__main__":
    main()
