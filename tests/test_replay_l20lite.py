"""Headless replay smoke test for the L20 Lite product paths.

Synthesises a tiny `telemetry.npz` matching the `data_collection`
column layout for L20 Lite hands (cols 0-13: arm radians, 14-33:
linker_l20lite byte commands, 10 per hand), drives it through
`run_replay` against the MuJoCo backend, and asserts the loop consumes
every frame without error.

Covers the two workstations the alignment layer unblocks once
`linker_l20lite` is registered: a7_lite_l20lite_dc and
p7_i1_l20lite_bimanual (both 7-DOF arms + 10-actuated L20 Lite hands).
Mirrors `test_replay_a7_lite.py`.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
mujoco = pytest.importorskip("mujoco")

from linker_sim.backends.mujoco.backend import MujocoBackendCfg, MujocoSimBackend
from linker_sim.io.replay.sources import TelemetryNpzSource
from linker_sim.runtime.replay import run_replay


# L20 Lite: 7-DOF arms + 10 actuated joints per hand. Byte columns feed
# the linker_l20lite decoder (0-255 -> joint range, same math as O6).
LAYOUT = {
    "arm_left":   {"cols": (0, 7),   "sign": 1.0},
    "arm_right":  {"cols": (7, 14),  "sign": 1.0},
    "hand_left":  {"cols": (14, 24), "decoder": "linker_l20lite"},
    "hand_right": {"cols": (24, 34), "decoder": "linker_l20lite"},
}
N_FRAMES = 5
N_COLS = 34


def _write_synthetic_npz(path):
    rng = np.random.default_rng(0)
    qpos = np.zeros((N_FRAMES, N_COLS), dtype=np.float32)
    qpos[:, 14:34] = rng.integers(0, 256, size=(N_FRAMES, 20)).astype(np.float32)
    np.savez(path, qpos=qpos)


@pytest.mark.parametrize("workstation", ["a7_lite_l20lite_dc", "p7_i1_l20lite_bimanual"])
def test_l20lite_replay_runs_to_completion(tmp_path, workstation):
    npz = tmp_path / "telemetry.npz"
    _write_synthetic_npz(npz)

    backend = MujocoSimBackend(MujocoBackendCfg(
        workstations={"robot": workstation},
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
