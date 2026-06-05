"""Headless replay smoke test for the Viser backend.

Mirrors `tests/test_replay_a7_lite.py`: synthesize a tiny telemetry
buffer matching the data_collection layout, drive it through
`run_replay()` against a headless ViserSimBackend, and assert the loop
consumes every frame without binding a websocket port.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
viser = pytest.importorskip("viser")
yourdfpy = pytest.importorskip("yourdfpy")

from linker_sim.backends.viser.backend import ViserBackendCfg, ViserSimBackend
from linker_sim.io.replay.sources import TelemetryNpzSource
from linker_sim.runtime.replay import run_replay


# Mirror packages/linker-sim/src/linker_sim/configs/source/data_collection.yaml.
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


def test_viser_replay_runs_headless(tmp_path):
    npz = tmp_path / "telemetry.npz"
    _write_synthetic_npz(npz)

    backend = ViserSimBackend(ViserBackendCfg(
        workstations={"robot": "a7_lite_o6_dc"},
        headless=True,
    ))
    # Lock in headless invariant: no websocket server, no scene.
    assert backend._server is None
    assert backend._viser_urdf is None
    try:
        robot = backend.robots["robot"]
        source = TelemetryNpzSource(path=tmp_path, layout=LAYOUT, hz=30.0)
        consumed = run_replay(
            backend, robot, source,
            realtime=False, max_frames=N_FRAMES, loop=False,
        )
        assert consumed == N_FRAMES
        assert torch.isfinite(robot.joint_pos).all()
        # Last-frame buffer should match the source's final decoded targets.
        last = source.joint_targets(N_FRAMES - 1)
        for role, expected in last.items():
            ids = robot.actuated_joint_ids_of(role).cpu().numpy()
            np.testing.assert_allclose(
                robot.joint_buffer[ids], expected, rtol=1e-5, atol=1e-5
            )
    finally:
        backend.close()
