"""Viser backend for `linker-sim`.

Replay-only browser visualization. Not a physics simulator — joint state
is driven directly from external telemetry. Useful for the data-collection
team who want to inspect bag replays on a workstation without a GPU.

See `packages/linker-sim/src/linker_sim/backends/viser/backend.py` and
`robot.py` for the implementation. Non-replay `Robot` protocol methods
(Jacobian, mass matrix, set_joint_effort, ee_pose_b) raise
NotImplementedError; teleop is deferred.
"""

from linker_sim.backends.viser.backend import ViserBackendCfg, ViserSimBackend

__all__ = ["ViserBackendCfg", "ViserSimBackend"]
