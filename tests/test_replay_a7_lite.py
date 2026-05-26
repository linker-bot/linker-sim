"""Headless replay smoke test for the a7_lite_o6_dc product path.

Synthesises a tiny `telemetry.npz` matching the `data_collection`
column layout (cols 0-13: arm radians, 14-25: linker_o6 byte
commands), drives it through `run_replay` against the MuJoCo
backend, and asserts the loop consumes every frame without error.

This is the only automated guard for the a7_lite_o6_dc + data
collection telemetry pairing — the older suite only covers
ar5_l6_bench_bimanual.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
mujoco = pytest.importorskip("mujoco")

from sim.backends.mujoco.backend import MujocoBackendCfg, MujocoSimBackend
from sim.io.replay.sources import TelemetryNpzSource
from sim.runtime.replay import run_replay


# Mirror sim/configs/source/data_collection.yaml. Kept in-test so a
# config rename can't silently turn this into a no-op.
LAYOUT = {
    "arm_left":  {"cols": (0, 7),   "sign": 1.0},
    "arm_right": {"cols": (7, 14),  "sign": 1.0},
    "hand_left":  {"cols": (14, 20), "decoder": "linker_o6"},
    "hand_right": {"cols": (20, 26), "decoder": "linker_o6"},
}
N_FRAMES = 5
N_COLS = 26


def _write_synthetic_npz(path):
    rng = np.random.default_rng(0)
    qpos = np.zeros((N_FRAMES, N_COLS), dtype=np.float32)
    qpos[:, 14:26] = rng.integers(0, 256, size=(N_FRAMES, 12)).astype(np.float32)
    np.savez(path, qpos=qpos)


def test_a7_lite_o6_dc_replay_runs_to_completion(tmp_path):
    npz = tmp_path / "telemetry.npz"
    _write_synthetic_npz(npz)

    backend = MujocoSimBackend(MujocoBackendCfg(
        workstations={"robot": "a7_lite_o6_dc"},
    ))
    try:
        robot = backend.robots["robot"]
        source = TelemetryNpzSource(path=tmp_path, layout=LAYOUT, hz=30.0)
        consumed = run_replay(
            backend, robot, source,
            realtime=False, max_frames=N_FRAMES, loop=False,
        )
        assert consumed == N_FRAMES
        assert torch.isfinite(robot.joint_pos).all()
    finally:
        backend.close()
