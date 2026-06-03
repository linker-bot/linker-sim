"""Convert data.json → telemetry.npz for the replay pipeline.

Usage:
    python3 scripts/convert_data_json.py data.json --out episode_json/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description="Convert data.json to telemetry.npz")
    p.add_argument("input", type=Path, help="path to data.json")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("episode_json"),
        help="output directory (telemetry.npz written here)",
    )
    args = p.parse_args()

    d = json.loads(args.input.read_text())
    hz = d["Hz"]
    left = np.array(d["left_robot_joint"], dtype=np.float32)
    right = np.array(d["right_robot_joint"], dtype=np.float32)
    qpos = np.concatenate([left, right], axis=1)
    T = qpos.shape[0]
    timestamps = np.arange(T, dtype=np.float64) / hz

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "telemetry.npz"
    np.savez(out_path, qpos=qpos, timestamps=timestamps)
    print(f"Wrote {out_path} — shape: {qpos.shape}, hz: {hz}")


if __name__ == "__main__":
    main()
