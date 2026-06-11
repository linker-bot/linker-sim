"""Nelder-Mead 6D anchor search for UMI replay.

Searches over (dx, dy, dz, anchor_roll, anchor_pitch, anchor_yaw) — the
preprocessor's anchor knobs — to minimize the mean position-tracking RMS
when the resulting trajectory is replayed through the full IK pipeline.

`--recenter` (centroid → anchor) is always on during the search; the 6D
delta is on top of recenter. Backend + IK controller are spawned once
and reused across iterations; per-eval cost is one warm-up + the full
frame loop with no plot/viewer.

Usage:
    PYTHONPATH=~/codes/UMI-Dex/src ~/opt/IsaacLab/env_isaaclab/bin/python \\
        scripts/anchor_search.py \\
        data/umi_episode_000001/capture_2026-05-29-11-29-42_ep001.bag \\
        --arm right --hz 30.0 --maxiter 80 \\
        --save-npz outputs/umi_replay/umi_ep1_searched.npz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[1]
for _src in ("packages/linker-sim/src", "packages/linker-robot-assets/src"):
    _abs = str(REPO_ROOT / _src)
    if _abs not in sys.path:
        sys.path.insert(0, _abs)

# TODO(linker-sim): replace with `from umi_dex...` once umi-dex is published
# to PyPI / internal index.
UMI_DEX_SRC = Path.home() / "codes" / "UMI-Dex" / "src"
if str(UMI_DEX_SRC) not in sys.path:
    sys.path.insert(0, str(UMI_DEX_SRC))

from scripts.umi_bag_to_ee_poses import (  # noqa: E402
    _pose_to_T,
    _read_vut_pose,
    _resample_pose,
    _T_to_pose7,
)


def _build_arm_pose(
    rebased: np.ndarray,
    T_ws_tool0: np.ndarray,
    x6: np.ndarray,
    *,
    recenter: bool = True,
    R_world: np.ndarray | None = None,
) -> np.ndarray:
    """Apply 6D anchor (translation + RPY) on top of rebased trajectory.

    If ``R_world`` is given, additionally rotate every pose's wrist
    orientation by R_world in workstation frame (positions untouched).
    """
    dx, dy, dz, ar, ap, ay = x6
    R_anchor = Rotation.from_euler("xyz", [ar, ap, ay]).as_matrix()
    T_anchor_extra = np.eye(4)
    T_anchor_extra[:3, :3] = R_anchor
    T_xyz = np.eye(4)
    T_xyz[:3, 3] = [dx, dy, dz]
    T_anchor = T_xyz @ T_ws_tool0 @ T_anchor_extra
    arm = np.stack([_T_to_pose7(T_anchor @ T) for T in rebased]).astype(np.float32)
    if recenter:
        anchor_pos = arm[0, :3].copy()
        centroid = arm[:, :3].mean(axis=0)
        arm[:, :3] += (anchor_pos - centroid)
    if R_world is not None:
        # Rotate orientations only; positions unchanged.
        quats_xyzw = arm[:, [4, 5, 6, 3]]
        rots_new = Rotation.from_matrix(R_world) * Rotation.from_quat(quats_xyzw)
        q_new_xyzw = rots_new.as_quat()
        arm[:, 3] = q_new_xyzw[:, 3].astype(np.float32)
        arm[:, 4:7] = q_new_xyzw[:, :3].astype(np.float32)
    return arm


def _eval_rms(
    backend,
    robot,
    ik,
    arm_role: str,
    arm_pose: np.ndarray,
    sub_steps: int,
    *,
    warmup_steps: int = 200,
) -> tuple[float, float]:
    """Run one IK replay pass, return (pos_rms_m, ori_rms_rad)."""
    n = len(arm_pose)
    robot.write_joint_state(robot.joint_pos_default, robot.joint_vel_default)

    target0 = torch.from_numpy(arm_pose[0]).unsqueeze(0)
    for _ in range(warmup_steps):
        ik.set_command(target0, robot)
        ik.apply(robot)
        for _ in range(sub_steps):
            backend.step()

    pos_errs = np.empty(n, dtype=np.float64)
    ori_errs = np.empty(n, dtype=np.float64)
    frame_name = f"{arm_role}:tool0"
    for t in range(n):
        ik.set_command(torch.from_numpy(arm_pose[t]).unsqueeze(0), robot)
        ik.apply(robot)
        for _ in range(sub_steps):
            backend.step()
        achieved = robot.ee_pose_b(frame_name)[0].numpy()
        pos_errs[t] = np.linalg.norm(arm_pose[t, :3] - achieved[:3])
        dot = abs(float(np.dot(arm_pose[t, 3:7], achieved[3:7])))
        ori_errs[t] = 2.0 * np.arccos(np.clip(dot, -1.0, 1.0))
    return float(np.sqrt(np.mean(pos_errs ** 2))), float(np.sqrt(np.mean(ori_errs ** 2)))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("bag", type=Path)
    p.add_argument("--arm", choices=["left", "right"], default="right")
    p.add_argument("--hz", type=float, default=30.0)
    p.add_argument("--workstation", default="a7_lite_dc")
    p.add_argument("--maxiter", type=int, default=80)
    p.add_argument(
        "--init", type=float, nargs=6, default=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        metavar=("DX", "DY", "DZ", "AR", "AP", "AY"),
        help="initial guess (m, m, m, rad, rad, rad). Default zeros, with --recenter on.",
    )
    p.add_argument(
        "--simplex-trans", type=float, default=0.05,
        help="initial-simplex translation step (m, default 5cm).",
    )
    p.add_argument(
        "--simplex-rot", type=float, default=np.deg2rad(5.0),
        help="initial-simplex rotation step (rad, default 5 deg).",
    )
    p.add_argument(
        "--warmup", type=int, default=200,
        help="IK warm-up iterations on frame 0 per eval (default 200, matches replay_ik).",
    )
    p.add_argument(
        "--max-frames", type=int, default=0,
        help="cap frames per eval (0 = full bag). Lower = faster search, less faithful.",
    )
    p.add_argument(
        "--mirror-x", action="store_true",
        help="reflect rebased trajectory about its YZ plane (flip x) before "
             "anchoring. Frame 0 stays identity; direction of progress and "
             "wrist x-axis flip. Re-run the search after toggling — the "
             "optimal anchor changes.",
    )
    p.add_argument(
        "--world-rotate-rpy", type=float, nargs=3, default=None,
        metavar=("R", "P", "Y"),
        help="rotate every pose's wrist orientation by RPY (rad, intrinsic "
             "XYZ) in workstation frame; positions unchanged. Use to redirect "
             "the palm. Example: palm +y → -z is `-1.5707 0 0` (−90° "
             "about workstation X). Re-run the search after toggling.",
    )
    p.add_argument("--save-npz", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    print(f"[anchor-search] reading {args.bag}", flush=True)
    t_pose_ns, poses = _read_vut_pose(args.bag)
    dt_ns = int(round(1e9 / args.hz))
    target_ns = np.arange(int(t_pose_ns[0]), int(t_pose_ns[-1]), dt_ns, dtype=np.int64)
    poses_rs = _resample_pose(t_pose_ns, poses, target_ns)
    if args.max_frames and args.max_frames < len(poses_rs):
        poses_rs = poses_rs[: args.max_frames]
    n_frames = len(poses_rs)
    print(f"[anchor-search] resampled to {n_frames} frames @ {args.hz}Hz", flush=True)

    # Rebase to frame-0 = identity.
    T0 = _pose_to_T(poses_rs[0, :3], poses_rs[0, 3:7])
    T0_inv = np.linalg.inv(T0)
    rebased = np.stack([T0_inv @ _pose_to_T(p[:3], p[3:7]) for p in poses_rs])

    if args.mirror_x:
        M = np.diag([-1.0, 1.0, 1.0, 1.0])
        rebased = np.stack([M @ T @ M for T in rebased])
        print("[anchor-search] --mirror-x: reflected rebased trajectory about YZ plane", flush=True)

    R_world = None
    if args.world_rotate_rpy is not None:
        R_world = Rotation.from_euler("xyz", args.world_rotate_rpy).as_matrix()
        deg = np.degrees(args.world_rotate_rpy)
        print(
            f"[anchor-search] --world-rotate-rpy: orientations pre-rotated in "
            f"workstation frame by ({deg[0]:+.2f}, {deg[1]:+.2f}, {deg[2]:+.2f}) deg",
            flush=True,
        )

    # Spawn backend once, attach IK.
    from linker_sim.backends.mujoco.backend import MujocoBackendCfg, MujocoSimBackend
    from linker_sim.controllers.ik import IkController, IkControllerCfg

    backend_dt = 1.0 / 500.0
    backend = MujocoSimBackend(
        MujocoBackendCfg(
            workstations={"robot": args.workstation},
            num_envs=1,
            dt=backend_dt,
            device="cpu",
        )
    )
    robot = backend.robots["robot"]
    arm_role = f"arm_{args.arm}"
    sub_steps = max(1, int(round(1.0 / (args.hz * backend_dt))))
    ee = robot.ee_pose_b(f"{arm_role}:tool0")[0].numpy()
    T_ws_tool0 = _pose_to_T(ee[:3], ee[3:7])

    ik = IkController(IkControllerCfg(role=arm_role, frame=f"{arm_role}:tool0", damping=0.05))
    ik.attach(robot)

    print(
        f"[anchor-search] backend ready: arm_role={arm_role}, sub_steps={sub_steps}, "
        f"warmup={args.warmup}, frames/eval={n_frames}",
        flush=True,
    )

    history: list[tuple[np.ndarray, float, float]] = []  # (x, pos_rms, ori_rms)
    eval_idx = [0]
    best = [float("inf"), None, None]  # rms, x, ori_rms

    def objective(x: np.ndarray) -> float:
        t0 = time.perf_counter()
        arm_pose = _build_arm_pose(rebased, T_ws_tool0, x, recenter=True, R_world=R_world)
        pos_rms, ori_rms = _eval_rms(
            backend, robot, ik, arm_role, arm_pose, sub_steps,
            warmup_steps=args.warmup,
        )
        eval_idx[0] += 1
        dt = time.perf_counter() - t0
        history.append((x.copy(), pos_rms, ori_rms))
        if pos_rms < best[0]:
            best[0], best[1], best[2] = pos_rms, x.copy(), ori_rms
            star = "*"
        else:
            star = " "
        print(
            f"[eval {eval_idx[0]:3d}{star}] "
            f"x=[{x[0]:+.3f},{x[1]:+.3f},{x[2]:+.3f}, "
            f"{np.degrees(x[3]):+5.1f},{np.degrees(x[4]):+5.1f},{np.degrees(x[5]):+5.1f}]deg "
            f"pos_rms={pos_rms*1000:6.2f}mm ori_rms={np.degrees(ori_rms):5.2f}deg "
            f"({dt:.2f}s)",
            flush=True,
        )
        return pos_rms

    x0 = np.asarray(args.init, dtype=np.float64)
    # Build mixed-units initial simplex explicitly so translation steps and
    # rotation steps each contribute commensurate-scale exploration.
    s_t, s_r = args.simplex_trans, args.simplex_rot
    steps = np.array([s_t, s_t, s_t, s_r, s_r, s_r])
    simplex = np.vstack([x0] + [x0 + np.eye(6)[i] * steps[i] for i in range(6)])

    print(f"[anchor-search] starting Nelder-Mead, maxiter={args.maxiter}", flush=True)
    t_search = time.perf_counter()
    result = minimize(
        objective,
        x0,
        method="Nelder-Mead",
        options={
            "maxiter": args.maxiter,
            "initial_simplex": simplex,
            "xatol": 1e-3,
            "fatol": 1e-4,
            "adaptive": True,
        },
    )
    t_total = time.perf_counter() - t_search

    print("\n" + "=" * 70)
    print(f"ANCHOR SEARCH RESULT  ({eval_idx[0]} evals, {t_total:.1f}s total)")
    print("=" * 70)
    print(f"  status: {result.message}  (success={result.success})")
    bx = best[1]
    print(
        f"  best x: dx={bx[0]:+.4f} dy={bx[1]:+.4f} dz={bx[2]:+.4f} m | "
        f"anchor_rpy=({np.degrees(bx[3]):+.2f}, {np.degrees(bx[4]):+.2f}, "
        f"{np.degrees(bx[5]):+.2f}) deg"
    )
    print(f"  best pos RMS: {best[0]*1000:.2f} mm")
    print(f"  ori RMS at best: {np.degrees(best[2]):.2f} deg")

    if args.save_npz is not None:
        arm_pose = _build_arm_pose(rebased, T_ws_tool0, np.asarray(bx), recenter=True, R_world=R_world)
        args.save_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.save_npz, **{f"arm_{args.arm}": arm_pose})
        print(f"  saved best trajectory: {args.save_npz}  shape={arm_pose.shape}")


if __name__ == "__main__":
    main()
