"""Hydra-driven entrypoint for the runtime backbone.

Usage:

    # Smoke rollout with OSC on a bimanual reach task:
    python scripts/run.py

    # MuJoCo + joint PD:
    python scripts/run.py backend=mujoco controller=joint_pd_bimanual task=bimanual_reach policy=zeros max_steps=200

    # IK-pose absolute control with JSONL recording:
    python scripts/run.py controller=ik_pose_bimanual task=bimanual_reach_ikpose recorder=jsonl

Hydra docs: https://hydra.cc/docs/intro/
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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


@hydra.main(config_path=str(REPO_ROOT / "sim" / "configs"), config_name="config", version_base="1.3")
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
    from sim.backends.mujoco.backend import MujocoBackendCfg, MujocoSimBackend

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
    from sim.backends.isaac.backend import IsaacBackendCfg, IsaacSimBackend

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
    from sim.envs.base import BaseEnv, BaseEnvCfg
    from sim.io.recorder import Recorder

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
    policy = _make_policy(cfg.policy, env.action_dim, backend.num_envs, backend.device, cfg=cfg, env=env)

    if hasattr(policy, "first_frame"):
        _teleport_to_first_frame(env, policy.first_frame)
        obs = env.task.observe(env.backend, env._last_action)

    max_steps = int(cfg.max_steps) if int(cfg.max_steps) > 0 else None
    if max_steps is None and getattr(policy, "num_frames", None) is not None:
        max_steps = int(policy.num_frames)
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

    realtime = bool(OmegaConf.select(cfg, "realtime", default=False))
    step_period = float(cfg.backend.dt) * int(cfg.decimation) if realtime else 0.0
    next_deadline = time.perf_counter() + step_period if realtime else 0.0

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
            if hasattr(policy, "first_frame"):
                _teleport_to_first_frame(env, policy.first_frame)
                obs = env.task.observe(env.backend, env._last_action)
            print(f"[run] manual reset at step {step}")
            if realtime:
                next_deadline = time.perf_counter() + step_period

        action = policy(step, obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if recorder is not None:
            recorder.record_step(obs, action, reward, terminated, truncated, info)
        if mj_viewer is not None:
            mj_viewer.sync()
        step += 1

        if realtime:
            sleep_for = next_deadline - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            next_deadline += step_period

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


def _make_policy(name: str, action_dim: int, num_envs: int, device, *, cfg=None, env=None):
    import torch

    if name == "zeros":
        def policy(_step, _obs):
            return torch.zeros((num_envs, action_dim), device=device)
    elif name == "random_walk":
        def policy(_step, _obs):
            return 0.1 * (2.0 * torch.rand((num_envs, action_dim), device=device) - 1.0)
    elif name == "replay":
        if cfg is None or env is None:
            raise ValueError("policy=replay requires cfg and env (called from _run_rollout)")
        return _make_replay_policy(cfg, env)
    else:
        raise ValueError(f"unknown policy: {name!r} (expected 'zeros', 'random_walk', or 'replay')")
    return policy


def _make_replay_policy(cfg, env):
    """Replay recorded arm trajectories through the joint-PD controllers.

    Telemetry layout is `[arm_left(7), arm_right(7), hand_left(6), hand_right(6)]`
    (verified empirically; cols 14-25 are 0-255 hand-command bytes, NOT joint
    angles). Hands are zeroed; arms are routed through the joint_pd_bimanual
    controller as `(target - default) / action_scale`. Errors out if the cmd
    would be clipped by `BaseEnv`'s `[-1, 1]` clamp — fix by overriding
    `controller.entries.{0,2}.cfg.action_scale`.
    """
    import numpy as np
    import torch

    rcfg = OmegaConf.select(cfg, "replay", default=None)
    if rcfg is None or OmegaConf.select(rcfg, "path", default=None) is None:
        raise SystemExit("error: policy=replay requires replay.path=<episode_dir_or_npz>")

    src = str(OmegaConf.select(rcfg, "source", default="qpos"))
    if src not in ("qpos", "actions"):
        raise SystemExit(f"error: replay.source must be 'qpos' or 'actions' (got {src!r})")
    swap = bool(OmegaConf.select(rcfg, "swap_arms", default=False))

    path = Path(rcfg.path)
    if path.is_dir():
        path = path / "telemetry.npz"
    if not path.is_file():
        raise SystemExit(f"error: replay file not found: {path}")
    data = np.load(path, allow_pickle=False)
    if src not in data.files:
        raise SystemExit(f"error: replay.source={src!r} not in {path} (keys: {list(data.files)})")
    arr = data[src]
    if arr.ndim != 2 or arr.shape[1] != 26:
        raise SystemExit(f"error: expected telemetry shape (T, 26), got {arr.shape}")
    n_frames = int(arr.shape[0])

    if swap:
        target_left = arr[:, 7:14].astype(np.float32)
        target_right = arr[:, 0:7].astype(np.float32)
    else:
        target_left = arr[:, 0:7].astype(np.float32)
        target_right = arr[:, 7:14].astype(np.float32)

    ctrls_by_role: dict = {}
    slices_by_role: dict = {}
    cursor = 0
    for c in env.controllers:
        role = getattr(c, "role", None) or getattr(getattr(c, "cfg", None), "role", None)
        if role is not None:
            ctrls_by_role[role] = c
            slices_by_role[role] = slice(cursor, cursor + c.command_dim)
        cursor += c.command_dim

    if "arm_left" not in ctrls_by_role or "arm_right" not in ctrls_by_role:
        raise SystemExit(
            "error: policy=replay requires arm_left and arm_right joint-PD entries "
            "(controller=joint_pd_bimanual)."
        )

    left_ctrl = ctrls_by_role["arm_left"]
    right_ctrl = ctrls_by_role["arm_right"]
    left_scale = float(left_ctrl.cfg.action_scale)
    right_scale = float(right_ctrl.cfg.action_scale)

    left_ids = env.robot.actuated_joint_ids_of("arm_left")
    right_ids = env.robot.actuated_joint_ids_of("arm_right")
    default = env.robot.joint_pos_default
    left_default = default[:, left_ids].cpu().numpy()
    right_default = default[:, right_ids].cpu().numpy()

    left_cmd = (target_left - left_default) / left_scale
    right_cmd = (target_right - right_default) / right_scale

    max_left = float(np.abs(left_cmd).max()) if left_cmd.size else 0.0
    max_right = float(np.abs(right_cmd).max()) if right_cmd.size else 0.0
    if max(max_left, max_right) > 1.0 + 1e-6:
        max_left_delta = float(np.abs(target_left - left_default).max())
        max_right_delta = float(np.abs(target_right - right_default).max())
        rec_left = max(round(max_left_delta * 1.1, 1), 1.0)
        rec_right = max(round(max_right_delta * 1.1, 1), 1.0)
        raise SystemExit(
            "error: replay command exceeds BaseEnv clamp [-1, 1] "
            f"(max |cmd|: left={max_left:.3f}, right={max_right:.3f}). "
            f"Override action_scale, e.g.: "
            f"controller.entries.0.cfg.action_scale={rec_left:.2f} "
            f"controller.entries.2.cfg.action_scale={rec_right:.2f}"
        )

    expected_dt = 1.0 / 30.0
    actual_dt = float(cfg.backend.dt) * int(cfg.decimation)
    if abs(actual_dt - expected_dt) / expected_dt > 0.05:
        print(
            f"[run][warn] env step dt = {actual_dt*1000:.2f} ms but telemetry is "
            f"30 Hz (~33.33 ms). Replay will run at {1/actual_dt:.1f} Hz. "
            f"Consider decimation=17 with backend.dt=1/500 (= 34 ms)."
        )

    left_t = torch.from_numpy(left_cmd)
    right_t = torch.from_numpy(right_cmd)
    template = torch.zeros((env.num_envs, env.action_dim), device=env.device)
    left_sl = slices_by_role["arm_left"]
    right_sl = slices_by_role["arm_right"]

    print(
        f"[run] replay: {n_frames} frames from {path.name} "
        f"(source={src}, swap_arms={swap}, max|cmd|: L={max_left:.3f} R={max_right:.3f})"
    )

    def policy(step, _obs):
        t = min(int(step), n_frames - 1)
        a = template.clone()
        a[:, left_sl] = left_t[t : t + 1].to(a.device)
        a[:, right_sl] = right_t[t : t + 1].to(a.device)
        return a

    policy.first_frame = {
        "arm_left": torch.from_numpy(target_left[0:1]).clone(),
        "arm_right": torch.from_numpy(target_right[0:1]).clone(),
    }
    policy.num_frames = n_frames
    return policy


def _teleport_to_first_frame(env, first_frame: dict) -> None:
    """Snap the robot to the first replay frame so the PD doesn't whip there."""
    jp = env.robot.joint_pos_default.clone()
    jv = env.robot.joint_vel_default.clone()
    for role, target in first_frame.items():
        ids = env.robot.actuated_joint_ids_of(role)
        jp[:, ids] = target.to(jp.device, dtype=jp.dtype)
    env.robot.write_joint_state(jp, jv)


if __name__ == "__main__":
    main()
