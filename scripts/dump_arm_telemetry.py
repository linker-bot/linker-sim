"""Extract arm columns from a telemetry.npz to CSV and plot the trajectory.

Telemetry layout (verified empirically): cols 0-6 = left arm (radians),
cols 7-13 = right arm, cols 14-25 = hand command bytes (skipped here).

Usage:
    python scripts/dump_arm_telemetry.py episode_000004
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


JOINT_NAMES = [f"L{i}" for i in range(1, 8)] + [f"R{i}" for i in range(1, 8)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episode", type=Path, help="episode dir or telemetry.npz path")
    parser.add_argument("--source", choices=["qpos", "actions"], default="qpos")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="output dir (default: alongside the npz)")
    args = parser.parse_args()

    npz = args.episode if args.episode.is_file() else args.episode / "telemetry.npz"
    data = np.load(npz, allow_pickle=False)
    arr = data[args.source]
    ts = data["timestamps"]
    if arr.shape[1] != 26:
        raise SystemExit(f"expected (T, 26), got {arr.shape}")

    arms = arr[:, :14]
    t_rel = ts - ts[0]

    out_dir = args.out_dir or npz.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"arms_{args.source}.csv"
    png_path = out_dir / f"arms_{args.source}.png"

    header = "t_s," + ",".join(JOINT_NAMES)
    np.savetxt(csv_path, np.column_stack([t_rel, arms]),
               delimiter=",", header=header, comments="", fmt="%.6f")
    print(f"wrote {csv_path} ({arr.shape[0]} rows)")

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for j in range(7):
        axes[0].plot(t_rel, arms[:, j], label=JOINT_NAMES[j])
        axes[1].plot(t_rel, arms[:, 7 + j], label=JOINT_NAMES[7 + j])
    axes[0].set_title(f"Left arm ({args.source})")
    axes[1].set_title(f"Right arm ({args.source})")
    for ax in axes:
        ax.set_ylabel("rad")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", ncols=7, fontsize=8)
    axes[1].set_xlabel("time (s)")
    fig.suptitle(f"{npz.parent.name} — arm joint trajectories ({arr.shape[0]} frames)")
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    print(f"wrote {png_path}")

    print("\nper-joint range (rad):")
    for j, name in enumerate(JOINT_NAMES):
        col = arms[:, j]
        print(f"  {name}: min={col.min():+.3f} max={col.max():+.3f} "
              f"std={col.std():.3f}")


if __name__ == "__main__":
    main()
