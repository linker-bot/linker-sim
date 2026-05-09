"""Protocol conformance + MuJoCo stub tests.

No Isaac / MuJoCo required — these tests cover the pure-Python contract
of `sim.backends.base` and the stubs under `sim.backends.mujoco`.
"""

from __future__ import annotations

import pytest

from sim.backends.base import Robot, SimBackend
from sim.backends.mujoco import MujocoBackendCfg, MujocoSimBackend


def test_mujoco_backend_raises_on_construct():
    with pytest.raises(NotImplementedError, match="PR #1b"):
        MujocoSimBackend(MujocoBackendCfg())


def test_protocol_runtime_checkable():
    # Both Protocols are runtime-checkable. A class that doesn't satisfy
    # them should isinstance-check False.
    class Dummy:
        pass

    assert not isinstance(Dummy(), Robot)
    assert not isinstance(Dummy(), SimBackend)


def test_protocol_structural_conformance():
    # A class that *does* satisfy the Protocol's attributes isinstance-checks True.
    # We test SimBackend because it has a small surface.
    import torch

    class FakeBackend:
        num_envs = 1
        device = torch.device("cpu")
        dt = 1 / 60
        robots: dict = {}
        rigid_bodies: dict = {}
        env_origins = torch.zeros(1, 3)

        def step(self): pass
        def write_data(self): pass
        def reset(self, env_ids=None): pass
        def close(self): pass

    assert isinstance(FakeBackend(), SimBackend)
