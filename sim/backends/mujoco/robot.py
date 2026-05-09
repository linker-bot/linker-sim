"""MuJoCo `Robot` stub. See `backend.py` for the unblock signal."""

from __future__ import annotations

from sim.registry import WorkstationHandle


_UNBLOCK_MSG = (
    "MuJoCo backend is not yet implemented. Blocked on PR #1b "
    "(component MJCF authoring)."
)


class MujocoRobot:
    handle: WorkstationHandle

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(_UNBLOCK_MSG)
