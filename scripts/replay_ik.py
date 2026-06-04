"""IK replay: drive robot arms to follow EE pose trajectories via IK.

Two input modes:
  - ee_poses: load pre-computed EE pose trajectories from an npz file.
    Expected format: npz with keys per arm role (e.g. "arm_left", "arm_right"),
    each a (T, 7) array of [x, y, z, qw, qx, qy, qz] in workstation-base frame.
  - from_qpos (default): compute FK from recorded joint telemetry (validation mode).
    Uses the existing replay source to extract qpos, teleports per frame, reads
    ee_pose_b to get ground-truth targets.

Convention: all EE poses are in the workstation-base frame with (w, x, y, z)
quaternion order (scalar-first). The workstation base is at world origin for
fixed-base robots.

TODO: add null-space joint-limit avoidance to the DLS solver to prevent
elbow/wrist wandering in the redundant DOF during long trajectories.

Usage:
    # Validation mode: FK from recorded qpos -> IK -> measure tracking error
    python scripts/replay_ik.py robot=a7_lite_o6_dc source=data_collection

    # EE pose input mode: replay from external pose file
    python scripts/replay_ik.py robot=a7_lite_o6_dc source=data_collection \
        ee_poses=/path/to/poses.npz

    # Limit frames
    python scripts/replay_ik.py robot=a7_lite_o6_dc source=data_collection max_frames=200
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "packages" / "linker-sim" / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "packages" / "linker-sim" / "src"))

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

OmegaConf.register_new_resolver("div", lambda a, b: a / b, replace=True)


def _load_ee_poses(
    path: str | Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load EE pose + optional hand-joint trajectories from npz.

    Keys starting with "arm_" → (T, 7) [x, y, z, qw, qx, qy, qz] in
    workstation-base frame. Keys starting with "hand_" → (T, N) joint
    angles in radians driven directly into the actuated hand joints.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ee_poses file not found: {path}")
    data = np.load(path, allow_pickle=False)
    arm_poses: dict[str, np.ndarray] = {}
    hand_targets: dict[str, np.ndarray] = {}
    for key in data.files:
        arr = data[key]
        if key.startswith("arm_"):
            if arr.ndim != 2 or arr.shape[1] != 7:
                raise ValueError(
                    f"ee_poses[{key!r}]: expected shape (T, 7), got {arr.shape}"
                )
            arm_poses[key] = arr.astype(np.float32)
        elif key.startswith("hand_"):
            if arr.ndim != 2:
                raise ValueError(
                    f"ee_poses[{key!r}]: expected shape (T, N), got {arr.shape}"
                )
            hand_targets[key] = arr.astype(np.float32)
        else:
            raise ValueError(
                f"ee_poses[{key!r}]: key must start with 'arm_' or 'hand_'"
            )
    if not arm_poses:
        raise ValueError(f"ee_poses {path}: no arm_* keys found")
    return arm_poses, hand_targets


def _compute_fk_targets(
    robot, source, arm_roles: list[str], n_frames: int
) -> dict[str, np.ndarray]:
    """Compute FK from recorded qpos for each frame. Returns (T, 7) per role."""
    fk_targets: dict[str, np.ndarray] = {
        role: np.zeros((n_frames, 7), dtype=np.float32) for role in arm_roles
    }
    for t in range(n_frames):
        targets = source.joint_targets(t)
        jp = robot.joint_pos_default.clone()
        jv = robot.joint_vel_default.clone()
        for role, tgt in targets.items():
            ids = robot.actuated_joint_ids_of(role)
            jp[:, ids] = torch.from_numpy(tgt).unsqueeze(0)
        robot.write_joint_state(jp, jv)

        for role in arm_roles:
            ee = robot.ee_pose_b(f"{role}:tool0")
            fk_targets[role][t] = ee[0].numpy()
    return fk_targets


def _run_ik_replay(
    backend,
    robot,
    ik_controllers: dict,
    ee_targets: dict[str, np.ndarray],
    arm_roles: list[str],
    sub_steps: int,
    n_frames: int,
    *,
    source=None,
    gt_joint_targets=None,
    hand_targets: dict[str, np.ndarray] | None = None,
    viewer=None,
    realtime: bool = False,
    hz: float = 30.0,
):
    """Run IK replay loop and collect tracking errors.

    In ee_poses mode (source=None), runs a warm-up phase to converge to the
    first target pose before starting measurement.
    """
    errors_pos: dict[str, list] = {r: [] for r in arm_roles}
    errors_ori: dict[str, list] = {r: [] for r in arm_roles}
    errors_joint: dict[str, list] | None = (
        {r: [] for r in arm_roles} if gt_joint_targets else None
    )

    # Teleport to a configuration near the first target.
    # In from_qpos mode, use the recorded joints; in ee_poses mode, warm up
    # by iterating IK on the first pose until the arms converge.
    jp = robot.joint_pos_default.clone()
    jv = robot.joint_vel_default.clone()
    if source is not None:
        first_targets = source.joint_targets(0)
        for role, tgt in first_targets.items():
            ids = robot.actuated_joint_ids_of(role)
            jp[:, ids] = torch.from_numpy(tgt).unsqueeze(0)
    robot.write_joint_state(jp, jv)

    if source is None:
        # Warm up: iterate IK on the first target until converged
        warmup_steps = 200
        for _ in range(warmup_steps):
            for role, ik in ik_controllers.items():
                target_pose = torch.from_numpy(ee_targets[role][0]).unsqueeze(0)
                ik.set_command(target_pose, robot)
            for role, ik in ik_controllers.items():
                ik.apply(robot)
            for _ in range(sub_steps):
                backend.step()
        print(f"[replay_ik] warm-up done ({warmup_steps} IK iterations on first frame)")
        if viewer is not None:
            viewer.sync()

    period = 1.0 / hz if hz > 0 else 0.0
    deadline = time.perf_counter() + period
    for t in range(n_frames):
        if viewer is not None and not viewer.is_running():
            print(f"[replay_ik] viewer closed; stopping at frame {t}")
            break

        # Set IK targets
        for role, ik in ik_controllers.items():
            target_pose = torch.from_numpy(ee_targets[role][t]).unsqueeze(0)
            ik.set_command(target_pose, robot)

        # Apply IK
        for role, ik in ik_controllers.items():
            ik.apply(robot)

        # Drive hands directly if source available
        if source is not None:
            hand_role_targets = source.joint_targets(t)
            for role, tgt in hand_role_targets.items():
                if role.startswith("hand_"):
                    ids = robot.actuated_joint_ids_of(role)
                    robot.set_joint_position_target(
                        torch.from_numpy(tgt).unsqueeze(0), ids
                    )
        elif hand_targets:
            for role, arr in hand_targets.items():
                ids = robot.actuated_joint_ids_of(role)
                robot.set_joint_position_target(
                    torch.from_numpy(arr[t]).unsqueeze(0), ids
                )

        # Step physics
        for _ in range(sub_steps):
            backend.step()

        if viewer is not None:
            viewer.sync()
        if realtime and period > 0:
            now = time.perf_counter()
            sleep_for = deadline - now
            if sleep_for > 0:
                time.sleep(sleep_for)
            deadline += period

        # Measure Cartesian errors
        for role in arm_roles:
            achieved_ee = robot.ee_pose_b(f"{role}:tool0")
            target_ee = ee_targets[role][t]
            pos_err = float(np.linalg.norm(
                target_ee[:3] - achieved_ee[0, :3].numpy()
            ))
            q_target = target_ee[3:7]
            q_achieved = achieved_ee[0, 3:7].numpy()
            dot = abs(np.dot(q_target, q_achieved))
            ori_err = 2.0 * np.arccos(np.clip(dot, -1.0, 1.0))
            errors_pos[role].append(pos_err)
            errors_ori[role].append(ori_err)

        # Joint-space error (only in validation mode)
        if errors_joint is not None and gt_joint_targets is not None:
            achieved_jp = robot.joint_pos
            for role in arm_roles:
                ids = robot.actuated_joint_ids_of(role)
                gt = torch.from_numpy(gt_joint_targets[role][t]).unsqueeze(0)
                achieved = achieved_jp[:, ids]
                errors_joint[role].append((gt - achieved).squeeze(0).numpy())

    return errors_pos, errors_ori, errors_joint


def _print_report(arm_roles, errors_pos, errors_ori, errors_joint):
    """Print tracking results to console."""
    print("\n" + "=" * 70)
    print("IK REPLAY TRACKING RESULTS")
    print("=" * 70)
    for role in arm_roles:
        pos_errs = np.array(errors_pos[role])
        ori_errs = np.array(errors_ori[role])

        print(f"\n--- {role} ---")
        print(
            f"  Cartesian position:  "
            f"RMS={np.sqrt(np.mean(pos_errs**2))*1000:.2f} mm  "
            f"Max={np.max(pos_errs)*1000:.2f} mm  "
            f"Mean={np.mean(pos_errs)*1000:.2f} mm"
        )
        print(
            f"  Orientation:         "
            f"RMS={np.degrees(np.sqrt(np.mean(ori_errs**2))):.3f} deg  "
            f"Max={np.degrees(np.max(ori_errs)):.3f} deg  "
            f"Mean={np.degrees(np.mean(ori_errs)):.3f} deg"
        )

        if errors_joint is not None and role in errors_joint:
            joint_errs = np.array(errors_joint[role])
            n_joints = joint_errs.shape[1]
            rms_per_joint = np.sqrt(np.mean(joint_errs ** 2, axis=0))
            max_per_joint = np.max(np.abs(joint_errs), axis=0)
            print(f"\n  Joint-space (validation):")
            print(f"  {'Joint':<8} {'RMS (deg)':<12} {'Max (deg)':<12}")
            for j in range(n_joints):
                print(
                    f"  J{j+1:<7} "
                    f"{np.degrees(rms_per_joint[j]):<12.3f} "
                    f"{np.degrees(max_per_joint[j]):<12.3f}"
                )


def _draw_trajectory_overlay(
    viewer,
    arm_targets: dict[str, np.ndarray],
    *,
    axis_every: int = 30,
    axis_len: float = 0.04,
    line_width: float = 2.0,
    axis_width: float = 3.5,
) -> None:
    """Draw each arm's (T, 7) trajectory into viewer.user_scn.

    Polyline through all positions in orange + RGB coordinate frames
    every ``axis_every`` samples (frame 0 and frame T-1 always drawn).
    Persistent — populated once, the viewer keeps rendering it across
    sync()s.
    """
    import mujoco
    from scipy.spatial.transform import Rotation

    scn = viewer.user_scn
    line_color = np.array([1.0, 0.55, 0.0, 1.0], dtype=np.float32)  # orange
    axis_colors = (
        np.array([1.0, 0.2, 0.2, 1.0], dtype=np.float32),  # x: red
        np.array([0.2, 1.0, 0.2, 1.0], dtype=np.float32),  # y: green
        np.array([0.3, 0.4, 1.0, 1.0], dtype=np.float32),  # z: blue
    )

    for poses in arm_targets.values():
        positions = poses[:, :3].astype(np.float64)
        quats_wxyz = poses[:, 3:7]
        # scipy expects xyzw
        rotations = Rotation.from_quat(quats_wxyz[:, [1, 2, 3, 0]])

        # Polyline.
        for i in range(len(positions) - 1):
            if scn.ngeom >= scn.maxgeom:
                return
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_connector(
                g,
                int(mujoco.mjtGeom.mjGEOM_LINE),
                line_width,
                positions[i],
                positions[i + 1],
            )
            g.rgba = line_color
            scn.ngeom += 1

        # Coordinate frames at every Nth waypoint, plus first and last.
        idxs = set(range(0, len(positions), axis_every))
        idxs.add(0)
        idxs.add(len(positions) - 1)
        for idx in sorted(idxs):
            R = rotations[idx].as_matrix()
            origin = positions[idx]
            for axis in range(3):
                if scn.ngeom >= scn.maxgeom:
                    return
                tip = origin + R[:, axis] * axis_len
                g = scn.geoms[scn.ngeom]
                mujoco.mjv_connector(
                    g,
                    int(mujoco.mjtGeom.mjGEOM_LINE),
                    axis_width,
                    origin,
                    tip,
                )
                g.rgba = axis_colors[axis]
                scn.ngeom += 1


def _save_plot(arm_roles, errors_pos, errors_ori):
    """Save tracking error timeseries as PNG."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[replay_ik] matplotlib not available; skipping plot.")
        return

    fig, axes = plt.subplots(len(arm_roles), 2, figsize=(14, 5 * len(arm_roles)))
    if len(arm_roles) == 1:
        axes = axes[np.newaxis, :]

    for i, role in enumerate(arm_roles):
        pos_errs = np.array(errors_pos[role]) * 1000
        ori_errs = np.degrees(np.array(errors_ori[role]))

        axes[i, 0].plot(pos_errs, linewidth=0.8)
        axes[i, 0].set_ylabel("Position error (mm)")
        axes[i, 0].set_xlabel("Frame")
        axes[i, 0].set_title(f"{role} — Cartesian position tracking error")
        axes[i, 0].grid(True, alpha=0.3)

        axes[i, 1].plot(ori_errs, linewidth=0.8, color="tab:orange")
        axes[i, 1].set_ylabel("Orientation error (deg)")
        axes[i, 1].set_xlabel("Frame")
        axes[i, 1].set_title(f"{role} — Orientation tracking error")
        axes[i, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = Path("ik_replay_tracking.png")
    plt.savefig(out_path, dpi=150)
    print(f"\n[replay_ik] plot saved: {out_path.resolve()}")


@hydra.main(
    config_path="pkg://linker_sim.configs",
    config_name="replay",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    print("[replay_ik] resolved cfg:\n" + OmegaConf.to_yaml(cfg), flush=True)

    from linker_sim.backends.mujoco.backend import MujocoBackendCfg, MujocoSimBackend
    from linker_sim.controllers.ik import IkController, IkControllerCfg
    from linker_sim.io.replay.sources import TelemetryNpzSource

    # --- Backend + robot ---
    backend = MujocoSimBackend(MujocoBackendCfg(
        workstations={cfg.robot.role_name: cfg.robot.workstation_name},
        num_envs=1,
        dt=float(cfg.backend.dt),
        device="cpu",
    ))
    robot = backend.robots[cfg.robot.role_name]

    # --- Determine input mode ---
    ee_poses_path = cfg.get("ee_poses", None)

    if ee_poses_path:
        # EE pose input mode: load directly from file
        print(f"[replay_ik] mode: ee_poses from {ee_poses_path}")
        ee_targets, hand_targets = _load_ee_poses(ee_poses_path)
        arm_roles = list(ee_targets)
        n_frames = min(
            min(v.shape[0] for v in ee_targets.values()),
            min((v.shape[0] for v in hand_targets.values()), default=10**9),
        )
        if cfg.max_frames:
            n_frames = min(n_frames, int(cfg.max_frames))
        source = None
        gt_joint_targets = None
        hz = float(cfg.get("hz", cfg.source.hz if "source" in cfg else 30.0))
        if hand_targets:
            print(
                f"[replay_ik] hand keys: "
                + ", ".join(f"{k}{v.shape}" for k, v in hand_targets.items())
            )
    else:
        # Validation mode: FK from recorded qpos
        print("[replay_ik] mode: from_qpos (validation)")
        source: TelemetryNpzSource = instantiate(cfg.source)
        source.bind_robot(robot)
        arm_roles = [r for r in source.roles if r.startswith("arm_")]
        n_frames = source.num_frames
        if cfg.max_frames:
            n_frames = min(n_frames, int(cfg.max_frames))
        hz = float(source.hz)
        hand_targets = None

        print("[replay_ik] computing FK from recorded qpos...")
        ee_targets = _compute_fk_targets(robot, source, arm_roles, n_frames)

        # Save ground-truth joint targets for joint-space error reporting
        gt_joint_targets = {
            role: np.array([source.joint_targets(t)[role] for t in range(n_frames)])
            for role in arm_roles
        }

    # --- Setup IK controllers ---
    sub_steps = max(1, int(round(1.0 / (hz * float(backend.dt)))))
    ik_controllers: dict[str, IkController] = {}
    for role in arm_roles:
        ik = IkController(IkControllerCfg(role=role, frame=f"{role}:tool0", damping=0.05))
        ik.attach(robot)
        ik_controllers[role] = ik

    print(
        f"[replay_ik] arms={arm_roles}, {n_frames} frames, "
        f"{sub_steps} sub-steps/frame (dt={backend.dt*1000:.1f}ms, hz={hz})"
    )

    # --- Run IK replay ---
    print("[replay_ik] running IK replay...")
    realtime = bool(cfg.get("realtime", False))
    headless = bool(cfg.get("headless", True))

    def _go(viewer):
        return _run_ik_replay(
            backend, robot, ik_controllers, ee_targets, arm_roles,
            sub_steps, n_frames,
            source=source,
            gt_joint_targets=gt_joint_targets,
            hand_targets=hand_targets,
            viewer=viewer,
            realtime=realtime,
            hz=hz,
        )

    if headless:
        errors_pos, errors_ori, errors_joint = _go(None)
    else:
        import mujoco.viewer as mjv

        with mjv.launch_passive(
            backend._model,
            backend._data,
            show_left_ui=False,
            show_right_ui=False,
        ) as viewer:
            _draw_trajectory_overlay(viewer, ee_targets)
            print(
                "[replay_ik] viewer launched; "
                f"trajectory overlay: {viewer.user_scn.ngeom} geoms "
                "(orange polyline + RGB axes). Close window to stop early."
            )
            errors_pos, errors_ori, errors_joint = _go(viewer)

    # --- Report + plot ---
    _print_report(arm_roles, errors_pos, errors_ori, errors_joint)
    _save_plot(arm_roles, errors_pos, errors_ori)


if __name__ == "__main__":
    main()
