"""Hand decoder convention test.

Locks in the linear-fit-v0 contract: SDK 0 → URDF lower limit, SDK 100 →
URDF upper limit, batched calls preserve shape and stay in [lo, hi].

Mirrors the test style in `tests/test_replay_a7_lite.py` (synthetic
input, direct import, no torch).
"""

from __future__ import annotations

import numpy as np
import pytest

from linker_robot_assets.decoders import CONVENTION, decode_hand


def test_convention_constant():
    assert CONVENTION == "linear-fit-v0"


@pytest.mark.parametrize(
    "name, side, n",
    [
        ("linkerhand_l6", "right", 6),
        ("linkerhand_l6", "left", 6),
        ("linkerhand_o6", "right", 6),
        ("linkerhand_o6", "left", 6),
        ("linkerhand_l25", "right", 16),
        ("linkerhand_l25", "left", 16),
    ],
)
def test_zero_returns_lower_limit(name, side, n):
    """Plan §4.1 done-when: decode_hand(name, side, np.zeros(n)) == URDF lower limits."""
    out = decode_hand(name, side, np.zeros(n))
    assert out.shape == (n,)
    assert out.dtype == np.float32
    # Pull the same limits via xml.etree directly to avoid trusting the
    # decoder's lookup as ground truth.
    import xml.etree.ElementTree as ET
    from pathlib import Path

    from linker_robot_assets import asset_root

    cdir = asset_root() / "components" / "hands" / name
    import yaml
    with (cdir / "decoder.yaml").open() as f:
        spec = yaml.safe_load(f)
    prefix = "l" if side == "left" else "r"
    joint_names = [c.replace("{S}", prefix) for c in spec["channels"]]
    tree = ET.parse(cdir / "variants" / side / "hand.urdf")
    by_name = {j.get("name"): j for j in tree.getroot().findall("joint")}
    lo = np.array(
        [float(by_name[j].find("limit").get("lower")) for j in joint_names],
        dtype=np.float32,
    )
    np.testing.assert_allclose(out, lo, atol=1e-5)


def test_full_returns_upper_limit():
    out = decode_hand("linkerhand_l6", "right", np.full(6, 100.0))
    # URDF upper limits for l6/right (verified separately).
    expected_hi = np.array(
        [1.256637, 0.837758, 1.134464, 1.134464, 1.134464, 1.134464],
        dtype=np.float32,
    )
    np.testing.assert_allclose(out, expected_hi, atol=1e-5)


def test_batched_shape_and_range():
    rng = np.random.default_rng(0)
    sdk = rng.uniform(0.0, 100.0, size=(5, 6)).astype(np.float32)
    out = decode_hand("linkerhand_o6", "right", sdk)
    assert out.shape == (5, 6)
    # All values in URDF [lo, hi].
    lo = decode_hand("linkerhand_o6", "right", np.zeros(6))
    hi = decode_hand("linkerhand_o6", "right", np.full(6, 100.0))
    assert (out >= lo - 1e-5).all()
    assert (out <= hi + 1e-5).all()


def test_clip_outside_0_100():
    """Values outside [0, 100] clip to the endpoints."""
    over = decode_hand("linkerhand_l6", "right", np.full(6, 150.0))
    hi = decode_hand("linkerhand_l6", "right", np.full(6, 100.0))
    np.testing.assert_allclose(over, hi, atol=1e-5)
    under = decode_hand("linkerhand_l6", "right", np.full(6, -10.0))
    lo = decode_hand("linkerhand_l6", "right", np.zeros(6))
    np.testing.assert_allclose(under, lo, atol=1e-5)


def test_channel_count_mismatch_raises():
    with pytest.raises(ValueError, match="channels"):
        decode_hand("linkerhand_l6", "right", np.zeros(7))  # l6 has 6
