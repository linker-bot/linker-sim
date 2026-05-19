"""Hand-encoding decoders.

Real-robot recordings often store hand commands in a sensor-native
encoding (Linker Hand uses 0–255 bytes per finger) rather than radians.
A decoder takes raw per-frame values plus the actuated-joint limits
for that hand role and returns joint-position targets in radians.
"""

from __future__ import annotations

from typing import Callable, Protocol

import numpy as np


class HandDecoder(Protocol):
    def __call__(
        self, raw: np.ndarray, lo: np.ndarray, hi: np.ndarray
    ) -> np.ndarray: ...


def linker_l6(raw: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Decode Linker Hand L6 0–255 byte commands to joint-position targets.

    TODO(real-calibration): the Linker Hand 0–255 scale corresponds to each
    finger's full mechanical travel from its lower joint limit to its upper
    joint limit; this implementation uses a plain linear mapping per joint.
    The vendor's real per-finger calibration may be non-linear and may
    invert travel direction on some joints. Replace this when the
    manufacturer curves are available.
    """
    raw = raw.astype(np.float32)
    return lo + (raw / 255.0) * (hi - lo)


_DECODERS: dict[str, Callable] = {
    "linker_l6": linker_l6,
}


def get(name: str) -> HandDecoder:
    if name not in _DECODERS:
        raise KeyError(f"unknown hand decoder {name!r} (have: {list(_DECODERS)})")
    return _DECODERS[name]
