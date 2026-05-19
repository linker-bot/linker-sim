"""Replay sources and hand decoders for external real-robot data.

A `ReplaySource` provides per-frame joint targets keyed by composer role
(arm_left, arm_right, hand_left, ...), already aligned to the workstation's
joint ordering. `sim.runtime.replay.run_replay` consumes this and drives the
sim — bypassing controllers, tasks, and BaseEnv entirely.

Concrete sources live in `sources.py`; decoders (Linker Hand bytes →
joint angles, etc.) live in `hands.py`.
"""

from sim.io.replay.sources import ReplaySource, RoleLayout, TelemetryNpzSource
from sim.io.replay import hands

__all__ = ["ReplaySource", "RoleLayout", "TelemetryNpzSource", "hands"]
