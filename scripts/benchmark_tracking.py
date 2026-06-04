"""Benchmark joint-position tracking accuracy in MuJoCo.

Replays real-robot telemetry through MuJoCo's PD actuators and measures
per-joint tracking error (commanded target vs achieved qpos after physics).

Usage:
    # Default: a7_lite_o6_dc, data_collection source
    python scripts/benchmark_tracking.py

    # Per-role gain overrides
    python scripts/benchmark_tracking.py gains.arm_left.stiffness=800 gains.arm_left.damping=6

    # Different workstation / source
    python scripts/benchmark_tracking.py robot=a7_lite_dc source=data_collection
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "packages" / "linker-sim" / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "packages" / "linker-sim" / "src"))

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

OmegaConf.register_new_resolver("div", lambda a, b: a / b, replace=True)


@hydra.main(config_path="pkg://linker_sim.configs", config_name="replay", version_base="1.3")
def main(cfg: DictConfig) -> None:
    from linker_sim.backends.mujoco.backend import MujocoBackendCfg, MujocoSimBackend

    source = instantiate(cfg.source)

    backend = MujocoSimBackend(MujocoBackendCfg(
        workstations={cfg.robot.role_name: cfg.robot.workstation_name},
        num_envs=1,
        dt=float(cfg.backend.dt),
        device="cpu",
    ))
    robot = backend.robots[cfg.robot.role_name]
    source.bind_robot(robot)

    gains_cfg = OmegaConf.select(cfg, "gains", default=None)
    if gains_cfg:
        for role, g in gains_cfg.items():
            kp = g.get("stiffness")
            kd = g.get("damping")
            if kp is not None and kd is not None:
                robot.write_gains(role, float(kp), float(kd))
                print(f"[benchmark] override gains {role}: kp={kp} kd={kd}")

    sub_steps = max(1, int(round(1.0 / (float(source.hz) * float(backend.dt)))))
    n_frames = source.num_frames
    max_frames = OmegaConf.select(cfg, "max_frames", default=None)
    if max_frames is not None and int(max_frames) > 0:
        n_frames = min(n_frames, int(max_frames))

    print(
        f"[benchmark] {source.describe()}\n"
        f"[benchmark] sub_steps={sub_steps} (dt={backend.dt*1000:.2f} ms), "
        f"frames={n_frames}"
    )

    # Teleport to first frame
    first_targets = source.joint_targets(0)
    jp = robot.joint_pos_default.clone()
    jv = robot.joint_vel_default.clone()
    for role, tgt in first_targets.items():
        ids = robot.actuated_joint_ids_of(role)
        jp[:, ids] = torch.from_numpy(tgt).unsqueeze(0)
    robot.write_joint_state(jp, jv)

    # Accumulate per-role targets and achieved positions
    role_targets: dict[str, list[np.ndarray]] = {r: [] for r in source.roles}
    role_achieved: dict[str, list[np.ndarray]] = {r: [] for r in source.roles}

    for t in range(n_frames):
        targets = source.joint_targets(t)
        for role, tgt in targets.items():
            ids = robot.actuated_joint_ids_of(role)
            tgt_t = torch.from_numpy(tgt).to(robot.device).unsqueeze(0)
            robot.set_joint_position_target(tgt_t, ids)

        for _ in range(sub_steps):
            backend.step()

        for role in source.roles:
            ids = robot.actuated_joint_ids_of(role)
            achieved = robot.joint_pos[:, ids].detach().cpu().numpy().flatten()
            role_targets[role].append(targets[role])
            role_achieved[role].append(achieved)

    # Compute and print metrics
    print("\n" + "=" * 72)
    print(f"{'TRACKING ERROR SUMMARY':^72}")
    print("=" * 72)

    role_errors: dict[str, np.ndarray] = {}
    for role in source.roles:
        tgt_arr = np.stack(role_targets[role])
        ach_arr = np.stack(role_achieved[role])
        err = tgt_arr - ach_arr
        role_errors[role] = err

        n_joints = err.shape[1]
        print(f"\n  Role: {role} ({n_joints} joints)")
        print(f"  {'joint':>8}  {'RMS(rad)':>10}  {'RMS(deg)':>10}  "
              f"{'MAE(rad)':>10}  {'MAX(rad)':>10}")
        print(f"  {'─' * 8}  {'─' * 10}  {'─' * 10}  {'─' * 10}  {'─' * 10}")
        for j in range(n_joints):
            col = err[:, j]
            rms = float(np.sqrt(np.mean(col ** 2)))
            mae = float(np.mean(np.abs(col)))
            mx = float(np.max(np.abs(col)))
            print(f"  {j:>8d}  {rms:>10.5f}  {np.degrees(rms):>10.3f}  "
                  f"{mae:>10.5f}  {mx:>10.5f}")

        total_rms = float(np.sqrt(np.mean(err ** 2)))
        print(f"  {'ALL':>8}  {total_rms:>10.5f}  {np.degrees(total_rms):>10.3f}")

    # Plot
    out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    png_path = out_dir / "tracking_benchmark.png"

    n_roles = len(source.roles)
    fig, axes = plt.subplots(n_roles * 2, 1, figsize=(12, 4 * n_roles), sharex=True)
    if n_roles * 2 == 1:
        axes = [axes]

    t_s = np.arange(n_frames) / source.hz

    for i, role in enumerate(source.roles):
        tgt_arr = np.stack(role_targets[role])
        ach_arr = np.stack(role_achieved[role])
        err = role_errors[role]
        n_joints = tgt_arr.shape[1]

        ax_overlay = axes[i * 2]
        ax_err = axes[i * 2 + 1]

        for j in range(n_joints):
            color = f"C{j}"
            ax_overlay.plot(t_s, tgt_arr[:, j], color=color, alpha=0.5, linewidth=0.8)
            ax_overlay.plot(t_s, ach_arr[:, j], color=color, linewidth=1.2,
                           linestyle="--", label=f"j{j}")
        ax_overlay.set_ylabel("rad")
        ax_overlay.set_title(f"{role} — target (solid) vs achieved (dashed)")
        ax_overlay.legend(loc="upper right", ncols=min(n_joints, 7), fontsize=7)
        ax_overlay.grid(True, alpha=0.3)

        for j in range(n_joints):
            ax_err.plot(t_s, np.abs(err[:, j]), linewidth=0.8, label=f"j{j}")
        ax_err.set_ylabel("|error| (rad)")
        ax_err.set_title(f"{role} — absolute tracking error")
        ax_err.legend(loc="upper right", ncols=min(n_joints, 7), fontsize=7)
        ax_err.grid(True, alpha=0.3)

    axes[-1].set_xlabel("time (s)")
    fig.suptitle(
        f"Tracking benchmark: {cfg.robot.workstation_name} — "
        f"{n_frames} frames @ {source.hz} Hz"
    )
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    print(f"\n[benchmark] plot saved: {png_path.resolve()}")


if __name__ == "__main__":
    main()
