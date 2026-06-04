"""Hydra-driven entrypoint for the runtime backbone.

Usage:

    # Smoke rollout with OSC on a bimanual reach task:
    python scripts/run.py

    # MuJoCo + joint PD:
    python scripts/run.py backend=mujoco controller=joint_pd_bimanual task=bimanual_reach policy=zeros max_steps=200

    # IK-pose absolute control with JSONL recording:
    python scripts/run.py controller=ik_pose_bimanual task=bimanual_reach_ikpose recorder=jsonl

For replaying real-robot data, use `scripts/replay.py` instead.

Hydra docs: https://hydra.cc/docs/intro/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "packages" / "linker-sim" / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "packages" / "linker-sim" / "src"))

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

# Register the div resolver used in configs (rationals for dt).
OmegaConf.register_new_resolver("div", lambda a, b: a / b, replace=True)


def _launch_isaac(cfg: DictConfig):
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(add_help=False)
    AppLauncher.add_app_launcher_args(parser)
    launch_args = parser.parse_args([])
    launch_args.headless = bool(cfg.headless)
    launch_args.device = str(cfg.device)
    app_launcher = AppLauncher(launch_args)
    return app_launcher.app


@hydra.main(config_path="pkg://linker_sim.configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print("[run] resolved cfg:\n" + OmegaConf.to_yaml(cfg), flush=True)

    if cfg.backend.name == "mujoco":
        _run_mujoco(cfg)
        return

    simulation_app = _launch_isaac(cfg)
    import traceback

    try:
        _run_isaac(cfg)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()


def _run_mujoco(cfg: DictConfig) -> None:
    from linker_sim.backends.mujoco.backend import MujocoBackendCfg, MujocoSimBackend

    if getattr(cfg.robot, "rigid_bodies", None):
        raise SystemExit(
            "error: rigid_bodies are not supported on the MuJoCo backend yet "
            "(use backend=isaac for pick_place)."
        )

    backend_cfg = MujocoBackendCfg(
        workstations={cfg.robot.role_name: cfg.robot.workstation_name},
        num_envs=int(cfg.num_envs),
        dt=float(cfg.backend.dt),
        device="cpu",
    )
    backend = MujocoSimBackend(backend_cfg)
    print(f"[run] backend ready: {cfg.robot.workstation_name} x{backend.num_envs}")

    if cfg.headless:
        _run_rollout(cfg, backend)
        return

    import mujoco.viewer as mjv

    reset_flag = [False]

    def on_key(keycode: int) -> None:
        if keycode in (ord("R"), ord("r")):
            reset_flag[0] = True

    with mjv.launch_passive(backend._model, backend._data, key_callback=on_key) as viewer:
        print("[run] hotkey: press 'R' in the viewport to reset")
        _run_rollout(cfg, backend, mj_viewer=viewer, mj_reset_flag=reset_flag)


def _run_isaac(cfg: DictConfig) -> None:
    from linker_sim.backends.isaac.backend import IsaacBackendCfg, IsaacSimBackend

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
    _run_rollout(cfg, backend, use_isaac_loop=True)


def _run_rollout(
    cfg: DictConfig,
    backend,
    *,
    use_isaac_loop: bool = False,
    mj_viewer=None,
    mj_reset_flag: list | None = None,
) -> None:
    from linker_sim.envs.base import BaseEnv, BaseEnvCfg
    from linker_sim.io.recorder import Recorder

    controllers = [instantiate(entry) for entry in cfg.controller.entries]
    task = instantiate(cfg.task, backend=backend)

    env_cfg = BaseEnvCfg(
        robot_name=cfg.robot.role_name,
        decimation=int(cfg.decimation),
        episode_length_s=float(cfg.episode_length_s),
        reset_joint_noise_scale=float(cfg.reset_joint_noise_scale),
    )
    env = BaseEnv(backend, controllers, task, env_cfg)
    print(f"[run] env ready: action_dim={env.action_dim} observation_dim={env.observation_dim}")

    recorder_cfg = OmegaConf.select(cfg, "recorder", default=None)
    recorder: Recorder | None = None
    if recorder_cfg is not None:
        recorder = instantiate(recorder_cfg, num_envs=backend.num_envs)

    obs, _ = env.reset(seed=int(cfg.seed))
    policy = _make_policy(cfg.policy, env.action_dim, backend.num_envs, backend.device)

    gain_watcher = None
    if bool(OmegaConf.select(cfg, "gain_tuner", default=False)):
        from pathlib import Path as _Path

        from linker_sim.io.gain_watcher import GainWatcher

        _gp = str(OmegaConf.select(cfg, "gain_tuner_path", default="/tmp/dex_pd_gains.json"))
        gain_watcher = GainWatcher(backend.robots[cfg.robot.role_name], _Path(_gp))

    max_steps = int(cfg.max_steps) if int(cfg.max_steps) > 0 else None
    if not use_isaac_loop and mj_viewer is None and max_steps is None:
        raise SystemExit(
            "error: backend=mujoco headless requires max_steps>0 (no viewport loop)."
        )

    if use_isaac_loop:
        reset_flag = _register_hotkey_reset(cfg.headless)
    elif mj_viewer is not None:
        reset_flag = mj_reset_flag if mj_reset_flag is not None else [False]
    else:
        reset_flag = [False]
    simulation_app = _get_simulation_app() if use_isaac_loop else None

    step = 0
    while True:
        if simulation_app is not None and not simulation_app.is_running():
            break
        if mj_viewer is not None and not mj_viewer.is_running():
            break
        if max_steps is not None and step >= max_steps:
            break

        if reset_flag[0]:
            reset_flag[0] = False
            obs, _ = env.reset()
            print(f"[run] manual reset at step {step}")

        if gain_watcher is not None:
            gain_watcher.tick()

        action = policy(step, obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if recorder is not None and action is not None:
            recorder.record_step(obs, action, reward, terminated, truncated, info)
        if mj_viewer is not None:
            mj_viewer.sync()
        step += 1

    if recorder is not None:
        recorder.close()
    print(f"[run] done after {step} steps.")


def _get_simulation_app():
    import omni.kit.app  # type: ignore

    return omni.kit.app.get_app()


def _register_hotkey_reset(headless: bool, key_name: str = "R") -> list:
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
    elif name == "hold":
        # Returns None — env.step skips controller writes so the GUI
        # (e.g. Isaac's Articulation Inspector / Gains Tuner) owns the
        # drive targets. Use for live gain tuning.
        def policy(_step, _obs):
            return None
    else:
        raise ValueError(f"unknown policy: {name!r} (expected 'zeros', 'random_walk', or 'hold')")
    return policy


if __name__ == "__main__":
    main()
