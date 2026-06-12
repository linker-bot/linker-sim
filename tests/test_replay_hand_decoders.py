"""Replay-byte hand decoder direction test.

Pins the convention of `linker_sim.io.replay.hands`:
    raw = scale -> joint at lo (rest / open)
    raw = 0     -> joint at hi (full travel / closed)

Uses synthetic per-channel limits — no robot, no backend — so the test
is direct: a regression here is a decoder bug, not a workstation
mismatch.
"""

from __future__ import annotations

import numpy as np
import pytest

from linker_sim.io.replay.hands import LinearByteDecoder, get


# Match real Linker Hand O6 right-hand limits — non-trivial values so a
# subtle off-by-one in the math wouldn't pass by coincidence.
LO = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
HI = np.array([1.36, 0.58, 1.60, 1.60, 1.60, 1.60], dtype=np.float32)


@pytest.mark.parametrize("name", ["linker_l6", "linker_o6", "linker_l25"])
def test_full_scale_byte_returns_lower_limit(name):
    """raw=255 -> joint at lo (rest / open hand)."""
    decoder = get(name)
    raw = np.full(LO.shape, 255.0, dtype=np.float32)
    out = decoder(raw, LO, HI)
    np.testing.assert_allclose(out, LO, atol=1e-5)


@pytest.mark.parametrize("name", ["linker_l6", "linker_o6", "linker_l25"])
def test_zero_byte_returns_upper_limit(name):
    """raw=0 -> joint at hi (full travel / closed hand)."""
    decoder = get(name)
    raw = np.zeros(LO.shape, dtype=np.float32)
    out = decoder(raw, LO, HI)
    np.testing.assert_allclose(out, HI, atol=1e-5)


def test_batched_shape_preserved():
    rng = np.random.default_rng(0)
    raw = rng.integers(0, 256, size=(7, 6)).astype(np.float32)
    out = get("linker_o6")(raw, LO, HI)
    assert out.shape == (7, 6)
    assert (out >= LO - 1e-5).all()
    assert (out <= HI + 1e-5).all()


def test_clips_outside_0_255():
    over = get("linker_o6")(np.full(6, 300.0, dtype=np.float32), LO, HI)
    at_full = get("linker_o6")(np.full(6, 255.0, dtype=np.float32), LO, HI)
    np.testing.assert_allclose(over, at_full, atol=1e-5)
    under = get("linker_o6")(np.full(6, -50.0, dtype=np.float32), LO, HI)
    at_zero = get("linker_o6")(np.zeros(6, dtype=np.float32), LO, HI)
    np.testing.assert_allclose(under, at_zero, atol=1e-5)


def test_linear_byte_decoder_inverted_flag():
    """A non-inverted decoder must map raw=scale -> hi (sanity check
    on the dataclass itself, not just the registered Linker rows)."""
    forward = LinearByteDecoder(scale=255.0, inverted=False)
    out = forward(np.full(6, 255.0, dtype=np.float32), LO, HI)
    np.testing.assert_allclose(out, HI, atol=1e-5)


def test_unknown_decoder_raises():
    with pytest.raises(KeyError):
        get("not_a_real_hand")
