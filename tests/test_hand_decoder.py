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
    the decoder's lookup as ground truth.

    Returns limits in the URDF's actuated-joint document order (non-fixed,
    non-mimic), which is the order decode_hand emits and the sim feeds
    positionally (== handle.joints[role])."""
    import xml.etree.ElementTree as ET

    from linker_robot_assets import asset_root

    cdir = asset_root() / "components" / "hands" / name
    # Tolerate both component layouts: variants/<side>/hand.urdf (l6/l25/
    # l20lite) and the flat <name>_<side>.urdf (o6). Mirrors
    # decoders.hand._resolve_hand_urdf.
    urdf = cdir / "variants" / side / "hand.urdf"
    if not urdf.is_file():
        urdf = cdir / f"{name}_{side}.urdf"
    tree = ET.parse(urdf)
    lo, hi = [], []
    for j in tree.getroot().findall("joint"):
        if j.get("type") == "fixed" or j.find("mimic") is not None:
            continue
        limit = j.find("limit")
        lo.append(float(limit.get("lower")))
        hi.append(float(limit.get("upper")))
    return np.array(lo, dtype=np.float32), np.array(hi, dtype=np.float32)


@pytest.mark.parametrize(
    "name, side, n",
    [
        ("linkerhand_l6", "right", 6),
        ("linkerhand_l6", "left", 6),
        ("linkerhand_o6", "right", 6),
        ("linkerhand_o6", "left", 6),
        ("linkerhand_l20lite", "right", 10),
        ("linkerhand_l20lite", "left", 10),
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
        ("linkerhand_l20lite", "right", 10),
        ("linkerhand_l20lite", "left", 10),
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


def test_legacy_channel_order_permutes_thumb_index_middle_pinky():
    """L6 legacy mapping differs from canonical only on the SDK channels
    that feed thumb_pitch / index / middle / pinky (thumb_roll and ring
    are shared). Output stays in manifest order either way.

    Canonical vs legacy joint <- SDK channel (see decoder.yaml):
        thumb_cmc_pitch : ch0 -> ch2
        index_mcp_pitch : ch2 -> ch0
        middle_mcp_pitch: ch3 -> ch5
        pinky_mcp_pitch : ch5 -> ch3
        thumb_cmc_roll  : ch1 (shared)
        ring_mcp_pitch  : ch4 (shared)
    """
    # Distinctive per-SDK-channel input so we can trace routing.
    sdk = np.arange(6, dtype=np.float32)[None, :] * 10.0
    canon = decode_hand("linkerhand_l6", "right", sdk)[0]
    legacy = decode_hand("linkerhand_l6", "right", sdk, legacy=True)[0]

    # Manifest (output) order: thumb_roll, thumb_pitch, index, middle, ring, pinky.
    # thumb_roll (idx 0, <-ch1) and ring (idx 4, <-ch4) unchanged.
    np.testing.assert_allclose(canon[0], legacy[0], atol=1e-6)  # thumb_roll
    np.testing.assert_allclose(canon[4], legacy[4], atol=1e-6)  # ring
    # thumb_pitch/index/middle/pinky each pick a different SDK channel.
    assert not np.isclose(canon[1], legacy[1])  # thumb_pitch: ch0 vs ch2
    assert not np.isclose(canon[2], legacy[2])  # index:       ch2 vs ch0
    assert not np.isclose(canon[3], legacy[3])  # middle:      ch3 vs ch5
    assert not np.isclose(canon[5], legacy[5])  # pinky:       ch5 vs ch3

    # Cross-check: legacy thumb_pitch == canonical value for SDK ch2 routed
    # into thumb_pitch, i.e. legacy index==canonical index swap is consistent.
    # thumb_pitch<-ch2 (legacy) equals index<-ch2 (canonical), same lo/hi? No —
    # different joints/limits, so just assert the routing via a fresh decode
    # where only ch2 is nonzero.
    only_ch2 = np.zeros((1, 6), dtype=np.float32)
    only_ch2[0, 2] = 100.0  # 100 -> lower limit for whichever joint ch2 feeds
    lg = decode_hand("linkerhand_l6", "right", only_ch2, legacy=True)[0]
    lo, _ = _urdf_limits("linkerhand_l6", "right", 6)
    # legacy ch2 -> thumb_cmc_pitch (manifest idx 1) should sit at its lower limit.
    np.testing.assert_allclose(lg[1], lo[1], atol=1e-5)


