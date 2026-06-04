"""Benchmark replay tracking accuracy.

Runs the same replay pipeline as `scripts/replay.py` but, instead of
just driving the sim, captures `robot.joint_pos[arm_ids]` after each
frame's physics step and compares it to the source's target at that
frame. Reports RMS error per joint, per role, and overall (arms only).

Usage:

    python scripts/benchmark_replay.py robot=ar5_o6_bench_bimanual \
        source=data_json backend=isaac

Honors the same hydra config as replay.py (`sim/configs/replay.yaml`).
Forces `headless=true`, `realtime=false`, `loop=false`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _src in ("packages/linker-sim/src", "packages/linker-robot-assets/src"):
    _abs = str(REPO_ROOT / _src)
    if _abs not in sys.path:
        sys.path.insert(0, _abs)

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

OmegaConf.register_new_resolver("div", lambda a, b: a / b, replace=True)

ARM_ROLES = ("arm_left", "arm_right")


@hydra.main(config_path="pkg://linker_sim.configs", config_name="replay", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print("[benchmark] resolved cfg:\n" + OmegaConf.to_yaml(cfg), flush=True)

    if cfg.backend.name == "mujoco":
        _benchmark_mujoco(cfg)
    else:
        _benchmark_isaac(cfg)


def _benchmark_mujoco(cfg: DictConfig) -> None:
    from linker_sim.backends.mujoco.backend import MujocoBackendCfg, MujocoSimBackend

    source = instantiate(cfg.source)
    backend = MujocoSimBackend(MujocoBackendCfg(
        workstations={cfg.robot.role_name: cfg.robot.workstation_name},
        num_envs=int(cfg.num_envs),
        dt=float(cfg.backend.dt),
        device="cpu",
    ))
    robot = backend.robots[cfg.robot.role_name]
    _run_and_report(backend, robot, source, cfg)


def _benchmark_isaac(cfg: DictConfig) -> None:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(add_help=False)
    AppLauncher.add_app_launcher_args(parser)
    launch_args = parser.parse_args([])
    launch_args.headless = True
    launch_args.device = str(cfg.device)
    app_launcher = AppLauncher(launch_args)
    simulation_app = app_launcher.app

    import traceback
    try:
        from linker_sim.backends.isaac.backend import IsaacBackendCfg, IsaacSimBackend

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
        _run_and_report(backend, robot, source, cfg)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()


def _run_and_report(backend, robot, source, cfg: DictConfig) -> None:
    source.bind_robot(robot)

    arm_roles = tuple(r for r in ARM_ROLES if r in source.roles)
    if not arm_roles:
        raise RuntimeError(
            f"none of {ARM_ROLES} present in source roles {source.roles!r}"
        )

    sub_steps = max(1, int(round(1.0 / (float(source.hz) * float(backend.dt)))))
    n_frames = source.num_frames if cfg.max_frames is None else min(
        source.num_frames, int(cfg.max_frames)
    )
    print(
        f"[benchmark] {source.describe()} -> "
        f"{sub_steps} physics steps per frame "
        f"(backend.dt={backend.dt*1000:.2f} ms), frames={n_frames}",
        flush=True,
    )

    arm_ids = {role: robot.actuated_joint_ids_of(role) for role in arm_roles}
    targets = {role: np.zeros((n_frames, int(arm_ids[role].numel())), np.float32)
               for role in arm_roles}
    observed = {role: np.zeros_like(targets[role]) for role in arm_roles}

    # Snap to first frame so the PD does not whip on startup.
    print("[benchmark] teleporting to frame 0...", flush=True)
    _teleport(robot, source.joint_targets(0))
    print("[benchmark] starting replay loop...", flush=True)

    log_every = max(1, n_frames // 10)
    for t in range(n_frames):
        frame_targets = source.joint_targets(t)
        for role, target in frame_targets.items():
            ids = robot.actuated_joint_ids_of(role)
            tgt = torch.from_numpy(target).to(robot.device).unsqueeze(0)
            robot.set_joint_position_target(tgt, ids)

        backend.write_data()
        for _ in range(sub_steps):
            backend.step()

        for role in arm_roles:
            ids = arm_ids[role]
            jp = robot.joint_pos[0, ids].detach().cpu().numpy()
            observed[role][t] = jp
            targets[role][t] = frame_targets[role]

        if (t + 1) % log_every == 0 or t == n_frames - 1:
            print(f"[benchmark] frame {t+1}/{n_frames}", flush=True)

    _report(targets, observed, arm_roles)


def _teleport(robot, first_frame: dict[str, np.ndarray]) -> None:
    jp = robot.joint_pos_default.clone()
    jv = robot.joint_vel_default.clone()
    for role, target in first_frame.items():
        ids = robot.actuated_joint_ids_of(role)
        tgt = torch.from_numpy(target).to(jp.device, dtype=jp.dtype)
        jp[:, ids] = tgt
        robot.set_joint_position_target(tgt.unsqueeze(0), ids)
    robot.write_joint_state(jp, jv)


def _report(targets, observed, arm_roles) -> None:
    print("\n========== Replay Tracking RMS (arms only) ==========")
    all_sq = []
    for role in arm_roles:
        diff = observed[role] - targets[role]
        per_joint_rms = np.sqrt((diff ** 2).mean(axis=0))   # (n_joints,)
        role_rms = float(np.sqrt((diff ** 2).mean()))
        all_sq.append(diff.reshape(-1) ** 2)

        print(f"\n[{role}] frames={diff.shape[0]} joints={diff.shape[1]}")
        print(f"  per-joint RMS (rad): "
              + ", ".join(f"j{i}={v:.5f}" for i, v in enumerate(per_joint_rms)))
        print(f"  per-joint RMS (deg): "
              + ", ".join(f"j{i}={np.degrees(v):.4f}" for i, v in enumerate(per_joint_rms)))
        print(f"  role mean RMS: {role_rms:.6f} rad ({np.degrees(role_rms):.4f} deg)")

    overall = float(np.sqrt(np.concatenate(all_sq).mean()))
    print(f"\n[overall arms] mean RMS: {overall:.6f} rad ({np.degrees(overall):.4f} deg)")
    print("======================================================")


if __name__ == "__main__":
    main()
