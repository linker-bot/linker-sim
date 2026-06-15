"""Hand-encoding decoders.

Real-robot recordings often store hand commands in a sensor-native
encoding (Linker Hand uses 0–255 bytes per finger) rather than radians.
A decoder takes raw per-frame values plus the actuated-joint limits
for that hand role and returns joint-position targets in radians.

The current Linker Hand byte convention (verified empirically against
Linker Hand O6 telemetry):

    raw = scale  -> rest / open pose -> joint at `lo`
    raw = 0      -> full travel      -> joint at `hi`

i.e. the byte rises with the rest pose, not with travel — `LinearByteDecoder`
calls this `inverted=True`. Each hand keeps its own row in the registry
even when the parameters match today, so per-hand calibration (per-finger
direction flips, manufacturer curves) can land on one hand without
touching the others.

TODO(real-calibration): linear travel is a placeholder. The vendor's
real per-finger response may be non-linear and may invert direction on
a subset of joints. When manufacturer curves are available, swap the
relevant row for a sibling decoder class (e.g. per-channel invert,
lookup table) — the `HandDecoder` protocol is the only contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class HandDecoder(Protocol):
    def __call__(
        self, raw: np.ndarray, lo: np.ndarray, hi: np.ndarray
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class LinearByteDecoder:
    """Linear byte → radians mapping with a global travel direction.

    `raw` is normalized to [0, 1] over `scale`, optionally inverted,
    then interpolated between (lo, hi). Values outside [0, scale] are
    clipped at the endpoints.
    """

    scale: float
    inverted: bool

    def __call__(
        self, raw: np.ndarray, lo: np.ndarray, hi: np.ndarray
    ) -> np.ndarray:
        norm = np.clip(raw.astype(np.float32) / self.scale, 0.0, 1.0)
        if self.inverted:
            norm = 1.0 - norm
        return lo + norm * (hi - lo)


# Per-hand registry. Rows are independent on purpose: when one hand's
# calibration diverges (e.g. a manufacturer curve, per-channel flips),
# only that row changes.
_DECODERS: dict[str, HandDecoder] = {
    "linker_l6":  LinearByteDecoder(scale=255.0, inverted=True),
    "linker_o6":  LinearByteDecoder(scale=255.0, inverted=True),
    "linker_l25": LinearByteDecoder(scale=255.0, inverted=True),
}


def get(name: str) -> HandDecoder:
    if name not in _DECODERS:
        raise KeyError(f"unknown hand decoder {name!r} (have: {list(_DECODERS)})")
    return _DECODERS[name]
