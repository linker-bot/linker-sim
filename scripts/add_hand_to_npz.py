"""Stitch hand telemetry from a UMI-Dex bag onto an existing arm-only npz.

The anchor-search pipeline writes only ``arm_<side>``; this script reads
``/hand/joint_states`` (already-decoded SDK percent, 0-100) from the
same bag, resamples to the arm npz's time grid, decodes to radians via
the workstation's hand-component decoder, and writes a combined npz
with both ``arm_<side>`` and ``hand_<side>`` keys.

Time-grid recovery: the arm npz holds ``n_frames`` poses sampled
uniformly at ``--hz`` starting from ``t_pose_ns[0]`` (the first
``/vut/pose`` message). This script reads ``/vut/pose`` just to pick
up that origin, then synthesizes the same grid.

Usage:
    PYTHONPATH=~/codes/UMI-Dex/src ~/opt/IsaacLab/env_isaaclab/bin/python \\
        scripts/add_hand_to_npz.py \\
        --bag data/umi_episode_000007/ \\
        --arm-npz outputs/umi_replay/umi_ep7_searched_mirrored_palmdown_warm9.npz \\
        --out outputs/umi_replay/umi_ep7_with_hand.npz \\
        --arm right --hz 30.0 --workstation a7_lite_l6_dc
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _src in (REPO_ROOT, REPO_ROOT / "packages/linker-sim/src",
             REPO_ROOT / "packages/linker-robot-assets/src"):
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

UMI_DEX_SRC = Path.home() / "codes" / "UMI-Dex" / "src"
if str(UMI_DEX_SRC) not in sys.path:
    sys.path.insert(0, str(UMI_DEX_SRC))


def _read_pose_origin_ns(bag_path: Path) -> int:
    from umi_dex.bag_reader import BagReader

    with BagReader(bag_path) as br:
        for sm in br.read_topic("/vut/pose"):
            return int(sm.t_ros_ns)
    raise RuntimeError(f"no /vut/pose messages in {bag_path}")


def _read_hand_joint_states(
    bag_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    from umi_dex.bag_reader import BagReader

    ts, pos, valid = [], [], []
    names: list[str] = []
    with BagReader(bag_path) as br:
        for sm in br.read_topic("/hand/joint_states"):
            ts.append(sm.t_ros_ns)
            pos.append(list(sm.msg.positions))
            valid.append(list(sm.msg.valid))
            if not names:
                names = list(sm.msg.names)
    if not pos:
        raise RuntimeError(f"no /hand/joint_states in {bag_path}")
    return (
        np.asarray(ts, dtype=np.int64),
        np.asarray(pos, dtype=np.float64),
        np.asarray(valid, dtype=bool),
        names,
    )


def _resample_nearest(
    t_src_ns: np.ndarray, vals: np.ndarray, t_tgt_ns: np.ndarray
) -> np.ndarray:
    idx = np.searchsorted(t_src_ns, t_tgt_ns, side="left")
    idx = np.clip(idx, 1, len(t_src_ns) - 1)
    left = idx - 1
    pick = np.where(
        np.abs(t_src_ns[idx] - t_tgt_ns) < np.abs(t_src_ns[left] - t_tgt_ns),
        idx, left,
    )
    return vals[pick]


def _hand_component(workstation: str, side: str) -> str:
    from linker_sim.backends.mujoco.backend import MujocoBackendCfg, MujocoSimBackend

    backend = MujocoSimBackend(MujocoBackendCfg(
        workstations={"robot": workstation}, num_envs=1,
        dt=1.0 / 500.0, device="cpu",
    ))
    ref = backend.robots["robot"].handle.components.get(f"hand_{side}")
    if ref is None:
        raise ValueError(f"workstation {workstation!r}: no hand_{side} component")
    return ref.name.split("/", 1)[-1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--bag", type=Path, required=True)
    p.add_argument("--arm-npz", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--arm", choices=["left", "right"], default="right")
    p.add_argument("--hz", type=float, default=30.0)
    p.add_argument("--workstation", default="a7_lite_l6_dc")
    args = p.parse_args()

    npz_in = np.load(args.arm_npz, allow_pickle=False)
    arm_key = f"arm_{args.arm}"
    if arm_key not in npz_in.files:
        raise KeyError(f"{args.arm_npz}: missing {arm_key!r} (have: {npz_in.files})")
    arm = npz_in[arm_key]
    n_frames = int(arm.shape[0])
    print(f"[stitch] arm npz: {arm_key} shape={arm.shape}", flush=True)

    t0_ns = _read_pose_origin_ns(args.bag)
    dt_ns = int(round(1e9 / args.hz))
    target_ns = t0_ns + np.arange(n_frames, dtype=np.int64) * dt_ns
    print(
        f"[stitch] time grid: {n_frames} frames @ {args.hz}Hz "
        f"from t_pose_ns[0]={t0_ns}",
        flush=True,
    )

    t_hand_ns, pct, valid, names = _read_hand_joint_states(args.bag)
    print(
        f"[stitch] /hand/joint_states: {len(pct)} msgs, names={names}, "
        f"valid_all={valid.all()}",
        flush=True,
    )
    pct_rs = _resample_nearest(t_hand_ns, pct, target_ns)
    static = [i for i in range(pct.shape[1]) if pct[:, i].ptp() < 1e-6]
    if static:
        print(
            f"[stitch] WARN: channels with no motion across the bag: "
            f"{[names[i] for i in static]} "
            f"(values: {[float(pct[0, i]) for i in static]})",
            flush=True,
        )

    component = _hand_component(args.workstation, args.arm)
    from linker_robot_assets.decoders import decode_hand, CONVENTION
    hand_rad = decode_hand(component, args.arm, pct_rs)
    print(
        f"[stitch] decoded via {component}/{args.arm} ({CONVENTION}): "
        f"hand shape={hand_rad.shape}, "
        f"rad range per ch: "
        + ", ".join(f"{hand_rad[:, i].min():+.2f}..{hand_rad[:, i].max():+.2f}"
                    for i in range(hand_rad.shape[1])),
        flush=True,
    )

    payload: dict[str, np.ndarray] = {arm_key: arm}
    payload[f"hand_{args.arm}"] = hand_rad.astype(np.float32)
    payload["decoder_convention"] = np.array(CONVENTION)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **payload)
    print(
        f"[stitch] wrote {args.out}: "
        + ", ".join(f"{k} {v.shape if hasattr(v, 'shape') else v}"
                    for k, v in payload.items()),
        flush=True,
    )


if __name__ == "__main__":
    main()
