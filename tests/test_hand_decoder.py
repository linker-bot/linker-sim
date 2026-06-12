"""Hand decoder convention test.

Locks in the linear-fit-v0 contract: SDK 100 → URDF lower limit (open),
SDK 0 → URDF upper limit (closed), batched calls preserve shape and
stay in [lo, hi].

Mirrors the test style in `tests/test_replay_a7_lite.py` (synthetic
input, direct import, no torch).
"""

from __future__ import annotations

import numpy as np
import pytest

from linker_robot_assets.decoders import CONVENTION, decode_hand


def test_convention_constant():
    assert CONVENTION == "linear-fit-v0"


def _urdf_limits(name, side, n):
    """Pull (lo, hi) URDF limits via xml.etree directly — avoid trusting
    the decoder's lookup as ground truth."""
    import xml.etree.ElementTree as ET
    from pathlib import Path

    import yaml

    from linker_robot_assets import asset_root

    cdir = asset_root() / "components" / "hands" / name
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
    hi = np.array(
        [float(by_name[j].find("limit").get("upper")) for j in joint_names],
        dtype=np.float32,
    )
    return lo, hi


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
def test_full_returns_lower_limit(name, side, n):
    """sdk=100 → URDF lower limit (rest / open hand)."""
    out = decode_hand(name, side, np.full(n, 100.0))
    assert out.shape == (n,)
    assert out.dtype == np.float32
    lo, _ = _urdf_limits(name, side, n)
    np.testing.assert_allclose(out, lo, atol=1e-5)


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
def test_zero_returns_upper_limit(name, side, n):
    """sdk=0 → URDF upper limit (full travel / closed hand)."""
    out = decode_hand(name, side, np.zeros(n))
    assert out.shape == (n,)
    assert out.dtype == np.float32
    _, hi = _urdf_limits(name, side, n)
    np.testing.assert_allclose(out, hi, atol=1e-5)


def test_batched_shape_and_range():
    rng = np.random.default_rng(0)
    sdk = rng.uniform(0.0, 100.0, size=(5, 6)).astype(np.float32)
    out = decode_hand("linkerhand_o6", "right", sdk)
    assert out.shape == (5, 6)
    # All values in URDF [lo, hi].
    lo, hi = _urdf_limits("linkerhand_o6", "right", 6)
    assert (out >= lo - 1e-5).all()
    assert (out <= hi + 1e-5).all()


def test_clip_outside_0_100():
    """Values outside [0, 100] clip to the endpoints."""
    over = decode_hand("linkerhand_l6", "right", np.full(6, 150.0))
    at_full = decode_hand("linkerhand_l6", "right", np.full(6, 100.0))
    np.testing.assert_allclose(over, at_full, atol=1e-5)
    under = decode_hand("linkerhand_l6", "right", np.full(6, -10.0))
    at_zero = decode_hand("linkerhand_l6", "right", np.zeros(6))
    np.testing.assert_allclose(under, at_zero, atol=1e-5)


def test_channel_count_mismatch_raises():
    with pytest.raises(ValueError, match="channels"):
        decode_hand("linkerhand_l6", "right", np.zeros(7))  # l6 has 6

