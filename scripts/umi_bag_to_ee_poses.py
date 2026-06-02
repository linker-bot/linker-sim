"""UMI-Dex bag → ee_poses npz preprocessor for replay_ik.py.

Reads a single UMI-Dex ROS1/ROS2 bag, extracts /vut/pose (VIVE Ultimate
Tracker 6DoF pose) and /hand/usart_raw (Linker glove encoder counts),
anchors the trajectory so frame 0 lands on the workstation's
default-pose tool0 FK, and writes an npz consumable by
`scripts/replay_ik.py` ee_poses mode.

Output keys (with --arm right):
- arm_right : (T, 7) float32 [x, y, z, qw, qx, qy, qz] in workstation
              base frame, anchored so frame 0 == FK(joint_pos_default)
              for arm_right:tool0 (modulo --dx/--dy/--dz/--dyaw nudge).
- hand_right: (T, 6) float32 joint angles in radians, mapped from
              UMI-Dex's per-finger calibration (raw counts → percent →
              radians via the workstation's hand joint limits).

Usage:
    python scripts/umi_bag_to_ee_poses.py <bag> \\
        --out outputs/umi_replay/umi_ep1.npz \\
        --arm right \\
        --hz 30.0 \\
        # Anchor (where the wrist starts, in workstation frame):
        [--dx 0 --dy 0 --dz 0]                # translation
        [--dyaw 0]                            # legacy: yaw whole trajectory about world Z
        [--anchor-roll 0 --anchor-pitch 0 --anchor-yaw 0]   # rotate ONLY frame 0 wrist
        # Tracker-axis remap (how recorded motion maps to world axes):
        [--remap-roll 0 --remap-pitch 0 --remap-yaw 0]      # rotates direction of progress only

Tuning workflow:
- Anchor knobs (`--anchor-*`) control the wrist's starting orientation.
- Remap knobs (`--remap-*`) rotate the trajectory's direction of progress
  while keeping frame 0 fixed (similarity: identity → identity).
- Use `--dx/--dy/--dz` to translate the entire trajectory in workstation
  frame. Frame 0 = (T_ws_tool0_default · R_anchor) + (dx, dy, dz).

KNOWN ISSUES (deferred — flag for the user, do not silently mask):
- /hand/usart_raw carries a per-channel `valid_mask` (bit i = channel i
  trustworthy). Some episodes drop bit 2 (index_pitch) for the opening
  frames. v0 ignores valid_mask and decodes whatever raw count is
  present; the resulting per-frame index-finger value will be junk
  during those windows. Revisit once the hand-firmware fix lands and
  future bags publish percent directly (then this whole calibrator path
  drops out — see docstring on _decode_hand below).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

UMI_DEX_SRC = Path.home() / "codes" / "UMI-Dex" / "src"
if str(UMI_DEX_SRC) not in sys.path:
    sys.path.insert(0, str(UMI_DEX_SRC))


# ---------- SE(3) helpers (Hamilton wxyz <-> scipy xyzw) ----------

def _wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return np.array([q[1], q[2], q[3], q[0]])


def _xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return np.array([q[3], q[0], q[1], q[2]])


def _pose_to_T(p: np.ndarray, q_wxyz: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(_wxyz_to_xyzw(q_wxyz)).as_matrix()
    T[:3, 3] = p
    return T


def _T_to_pose7(T: np.ndarray) -> np.ndarray:
    p = T[:3, 3]
    q_xyzw = Rotation.from_matrix(T[:3, :3]).as_quat()
    return np.concatenate([p, _xyzw_to_wxyz(q_xyzw)])


# ---------- bag readers ----------

def _read_vut_pose(bag_path: Path) -> tuple[np.ndarray, np.ndarray]:
    from umi_dex.bag_reader import BagReader

    ts: list[int] = []
    rows: list[list[float]] = []
    with BagReader(bag_path) as br:
        for sm in br.read_topic("/vut/pose"):
            p, q = sm.msg.pose.position, sm.msg.pose.orientation
            ts.append(sm.t_ros_ns)
            rows.append([p.x, p.y, p.z, q.w, q.x, q.y, q.z])
    if not rows:
        raise RuntimeError(f"no /vut/pose messages in {bag_path}")
    return np.asarray(ts, dtype=np.int64), np.asarray(rows, dtype=np.float64)


def _read_hand_raw(bag_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from umi_dex.bag_reader import BagReader

    ts: list[int] = []
    counts: list[list[int]] = []
    masks: list[int] = []
    with BagReader(bag_path) as br:
        for sm in br.read_topic("/hand/usart_raw"):
            ts.append(sm.t_ros_ns)
            counts.append(list(sm.msg.raw))
            masks.append(int(sm.msg.valid_mask))
    if not counts:
        raise RuntimeError(f"no /hand/usart_raw messages in {bag_path}")
    return (
        np.asarray(ts, dtype=np.int64),
        np.asarray(counts, dtype=np.float64),
        np.asarray(masks, dtype=np.int64),
    )


# ---------- resampling ----------

def _resample_pose(
    t_src_ns: np.ndarray, poses: np.ndarray, t_tgt_ns: np.ndarray
) -> np.ndarray:
    """Linear-interp position + Slerp orientation onto t_tgt_ns. Returns (T, 7)."""
    t = (t_src_ns - t_src_ns[0]).astype(np.float64) / 1e9
    t_tgt = (t_tgt_ns - t_src_ns[0]).astype(np.float64) / 1e9
    pos = np.stack([np.interp(t_tgt, t, poses[:, i]) for i in range(3)], axis=1)
    rots = Rotation.from_quat(np.stack([_wxyz_to_xyzw(q) for q in poses[:, 3:7]]))
    slerp = Slerp(t, rots)
    rt = slerp(np.clip(t_tgt, t.min(), t.max()))
    quats = np.stack([_xyzw_to_wxyz(q) for q in rt.as_quat()])
    return np.concatenate([pos, quats], axis=1)


def _resample_nearest(
    t_src_ns: np.ndarray, vals: np.ndarray, t_tgt_ns: np.ndarray
) -> np.ndarray:
    """Nearest-neighbor — avoids interpolating raw counts across wrap."""
    idx = np.searchsorted(t_src_ns, t_tgt_ns, side="left")
    idx = np.clip(idx, 1, len(t_src_ns) - 1)
    left = idx - 1
    pick = np.where(
        np.abs(t_src_ns[idx] - t_tgt_ns) < np.abs(t_src_ns[left] - t_tgt_ns),
        idx,
        left,
    )
    return vals[pick]


# ---------- workstation FK + hand limits ----------

def _spawn_workstation_anchors(
    workstation_name: str, arm_side: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (T_ws←tool0_default, hand_lo (rad), hand_hi (rad))."""
    from sim.backends.mujoco.backend import MujocoBackendCfg, MujocoSimBackend

    backend = MujocoSimBackend(
        MujocoBackendCfg(
            workstations={"robot": workstation_name},
            num_envs=1,
            dt=1.0 / 500.0,
            device="cpu",
        )
    )
    robot = backend.robots["robot"]
    robot.write_joint_state(robot.joint_pos_default, robot.joint_vel_default)
    ee = robot.ee_pose_b(f"arm_{arm_side}:tool0")[0].numpy()
    T_ws_tool0 = _pose_to_T(ee[:3], ee[3:7])
    lo, hi = robot.actuated_joint_limits_of(f"hand_{arm_side}")
    return T_ws_tool0, lo.numpy().astype(np.float64), hi.numpy().astype(np.float64)


# ---------- hand decoding ----------

def _decode_hand(raw_counts: np.ndarray, hand_lo: np.ndarray, hand_hi: np.ndarray) -> np.ndarray:
    """Decode raw 12-bit ADC counts → joint radians.

    Pipeline: counts → percent_open (0..100) via UMI-Dex's per-finger
    calibrator (handles thumb_roll wrap) → joint radians via:

        joint = lo + (1 - percent_open/100) * (hi - lo)

    convention matches existing `linker_o6` decoder (raw=open → joint=lo,
    raw=closed → joint=hi).

    Future bags will publish percent directly; when that lands, this
    function collapses to the linear map only — drop the Calibrator
    import.
    """
    from umi_dex.controllers.calibrate import Calibrator

    calib = Calibrator()
    pct = np.array(
        [calib.map_counts(row.tolist()) for row in raw_counts], dtype=np.float64
    )
    return (hand_lo + (1.0 - pct / 100.0) * (hand_hi - hand_lo)).astype(np.float32)


# ---------- main ----------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("bag", type=Path, help="UMI bag (ROS1 .bag or ROS2 dir)")
    p.add_argument("--out", type=Path, required=True, help="output .npz path")
    p.add_argument("--arm", choices=["left", "right"], default="right")
    p.add_argument("--hz", type=float, default=30.0)
    # Anchor translation (workstation base frame).
    p.add_argument("--dx", type=float, default=0.0)
    p.add_argument("--dy", type=float, default=0.0)
    p.add_argument("--dz", type=float, default=0.0)
    # Anchor orientation (extra rotation pre-multiplied into tool0_default).
    # These set what the wrist points at on frame 0; they do NOT affect
    # the trajectory's direction of progress.
    p.add_argument(
        "--anchor-roll", type=float, default=0.0,
        help="extra roll (rad) on anchor orientation, intrinsic XYZ",
    )
    p.add_argument(
        "--anchor-pitch", type=float, default=0.0,
        help="extra pitch (rad) on anchor orientation, intrinsic XYZ",
    )
    p.add_argument(
        "--anchor-yaw", type=float, default=0.0,
        help="extra yaw (rad) on anchor orientation, intrinsic XYZ. "
             "Combined: R_anchor = Rz(yaw)·Ry(pitch)·Rx(roll)",
    )
    p.add_argument(
        "--dyaw", type=float, default=0.0,
        help="legacy: yaw (rad) about workstation +Z applied OUTSIDE the "
             "anchor (rotates whole trajectory including frame 0). Prefer "
             "--anchor-yaw + --remap-yaw if you want to decouple.",
    )
    # Tracker-axis remap. Rotates the per-frame deltas via similarity:
    # delta' = R_remap · delta · R_remap^T. Frame 0 (identity) is fixed
    # under similarity, so the wrist start pose is unchanged; only the
    # direction of progress rotates.
    p.add_argument(
        "--remap-roll", type=float, default=0.0,
        help="rotate tracker-delta axes (rad), intrinsic XYZ",
    )
    p.add_argument(
        "--remap-pitch", type=float, default=0.0,
        help="rotate tracker-delta axes (rad), intrinsic XYZ",
    )
    p.add_argument(
        "--remap-yaw", type=float, default=0.0,
        help="rotate tracker-delta axes (rad), intrinsic XYZ",
    )
    p.add_argument("--workstation", default="a7_lite_dc")
    p.add_argument(
        "--no-hand", action="store_true",
        help="skip /hand/usart_raw entirely; write only arm_<side>. Use "
             "when the glove telemetry is unusable (e.g. railed channels).",
    )
    p.add_argument(
        "--recenter", action="store_true",
        help="translate the trajectory so its position centroid lands on "
             "the anchor point, instead of frame 0. Keeps a long trajectory "
             "inside the arm workspace (excursions become symmetric about "
             "the anchor). Orientation anchoring is unchanged.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    print(f"[umi-prep] reading {args.bag}", flush=True)
    t_pose_ns, poses = _read_vut_pose(args.bag)
    if args.no_hand:
        print("[umi-prep] --no-hand: skipping /hand/usart_raw", flush=True)
        t_hand_ns = raw_counts = masks = None
        print(
            f"[umi-prep] /vut/pose: {len(poses)} msgs "
            f"({(t_pose_ns[-1] - t_pose_ns[0]) / 1e9:.2f}s)",
            flush=True,
        )
    else:
        t_hand_ns, raw_counts, masks = _read_hand_raw(args.bag)
        print(
            f"[umi-prep] /vut/pose: {len(poses)} msgs "
            f"({(t_pose_ns[-1] - t_pose_ns[0]) / 1e9:.2f}s); "
            f"/hand/usart_raw: {len(raw_counts)} msgs",
            flush=True,
        )
        invalid_chans = ((~masks) & 0x3F)
        n_partial = int(np.count_nonzero(invalid_chans))
        if n_partial:
            # bit i unset → channel i invalid. Order: 0=thumb_roll,
            # 1=thumb_pitch, 2=index_pitch, 3=middle_pitch, 4=ring_pitch,
            # 5=pinky_pitch.
            bad_bits = np.bincount(
                np.concatenate([
                    np.where([bool(m & (1 << i)) is False for i in range(6)])[0]
                    for m in masks
                ]),
                minlength=6,
            )
            print(
                f"[umi-prep] WARN: {n_partial}/{len(masks)} hand frames have "
                f"≥1 invalid channel (per-channel drop counts: "
                f"thumb_roll={bad_bits[0]}, thumb_pitch={bad_bits[1]}, "
                f"index_pitch={bad_bits[2]}, middle_pitch={bad_bits[3]}, "
                f"ring_pitch={bad_bits[4]}, pinky_pitch={bad_bits[5]}). "
                "v0 ignores valid_mask — invalid channels decode to junk for "
                "those frames. Address in a follow-up.",
                flush=True,
            )

    # Sync window + uniform target timestamps at --hz.
    t0_ns = int(t_pose_ns[0])
    t1_ns = int(t_pose_ns[-1])
    if not args.no_hand:
        t0_ns = max(t0_ns, int(t_hand_ns[0]))
        t1_ns = min(t1_ns, int(t_hand_ns[-1]))
    dt_ns = int(round(1e9 / args.hz))
    target_ns = np.arange(t0_ns, t1_ns, dt_ns, dtype=np.int64)
    if len(target_ns) < 2:
        raise RuntimeError(
            f"sync window too short: {(t1_ns - t0_ns) / 1e9:.3f}s @ {args.hz}Hz"
        )

    poses_rs = _resample_pose(t_pose_ns, poses, target_ns)
    counts_rs = (
        None if args.no_hand
        else _resample_nearest(t_hand_ns, raw_counts, target_ns)
    )

    # Anchor frame 0 of /vut/pose to workstation tool0_default.
    # Trajectory in tracker frame, rebased so frame 0 = identity.
    T0 = _pose_to_T(poses_rs[0, :3], poses_rs[0, 3:7])
    T0_inv = np.linalg.inv(T0)
    rebased = np.stack([T0_inv @ _pose_to_T(p[:3], p[3:7]) for p in poses_rs])

    # Tracker-axis remap: similarity transform rotates direction of
    # progress while leaving identity (frame 0) fixed.
    R_remap = Rotation.from_euler(
        "xyz",
        [args.remap_roll, args.remap_pitch, args.remap_yaw],
    ).as_matrix()
    T_remap = np.eye(4); T_remap[:3, :3] = R_remap
    T_remap_inv = np.eye(4); T_remap_inv[:3, :3] = R_remap.T
    rebased = np.stack([T_remap @ d @ T_remap_inv for d in rebased])

    T_ws_tool0, hand_lo, hand_hi = _spawn_workstation_anchors(
        args.workstation, args.arm
    )
    # Anchor: translate, then rotate (anchor orientation), then yaw the
    # whole composed anchor (legacy --dyaw), all in workstation frame.
    R_anchor_extra = Rotation.from_euler(
        "xyz",
        [args.anchor_roll, args.anchor_pitch, args.anchor_yaw],
    ).as_matrix()
    T_anchor_extra = np.eye(4); T_anchor_extra[:3, :3] = R_anchor_extra
    T_xyz = np.eye(4); T_xyz[:3, 3] = [args.dx, args.dy, args.dz]
    T_dyaw = np.eye(4)
    T_dyaw[:3, :3] = Rotation.from_euler("z", args.dyaw).as_matrix()
    T_anchor = T_xyz @ T_dyaw @ T_ws_tool0 @ T_anchor_extra
    arm_pose = np.stack([_T_to_pose7(T_anchor @ T) for T in rebased]).astype(
        np.float32
    )

    # Recenter: translate so the position centroid lands on the anchor
    # point (where frame 0 sits), keeping a long trajectory inside the arm
    # workspace. Pure translation — orientation is untouched.
    if args.recenter:
        anchor_pos = arm_pose[0, :3].copy()
        centroid = arm_pose[:, :3].mean(axis=0)
        shift = (anchor_pos - centroid).astype(np.float32)
        arm_pose[:, :3] += shift
        print(
            f"[umi-prep] --recenter: shifted trajectory by "
            f"[{shift[0]:+.3f}, {shift[1]:+.3f}, {shift[2]:+.3f}] m "
            "to put centroid on anchor",
            flush=True,
        )

    hand_rad = None if args.no_hand else _decode_hand(counts_rs, hand_lo, hand_hi)

    # Verify invariants.
    assert arm_pose.shape == (len(target_ns), 7), arm_pose.shape
    if not args.no_hand:
        assert hand_rad.shape == (len(target_ns), 6), hand_rad.shape
    q0_norm = float(np.linalg.norm(arm_pose[0, 3:7]))
    assert abs(q0_norm - 1.0) < 1e-4, f"non-unit q0: {q0_norm}"
    no_anchor_translate = args.dx == args.dy == args.dz == 0.0
    no_anchor_yaw = args.dyaw == 0.0
    if no_anchor_translate and no_anchor_yaw and not args.recenter:
        # Anchor position is invariant under remap and anchor-orientation
        # rotations (those only rotate, never translate).
        p0_diff = float(np.linalg.norm(arm_pose[0, :3] - T_ws_tool0[:3, 3]))
        assert p0_diff < 1e-4, (
            f"frame-0 anchor position mismatch: {p0_diff}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {f"arm_{args.arm}": arm_pose}
    if not args.no_hand:
        payload[f"hand_{args.arm}"] = hand_rad
    np.savez(args.out, **payload)
    shapes = ", ".join(f"{k} {v.shape}" for k, v in payload.items())
    print(f"[umi-prep] wrote {args.out}: {shapes}", flush=True)


if __name__ == "__main__":
    main()
