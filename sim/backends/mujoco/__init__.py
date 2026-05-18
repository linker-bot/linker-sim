"""MuJoCo backend — `MujocoSimBackend` + `MujocoRobot`.

Loads composed `workstation.mjcf` artifacts from `sim.registry`.
B=1 per process (D8); numpy↔torch conversion at the robot boundary (D10).
"""

from __future__ import annotations

from sim.backends.mujoco.backend import MujocoBackendCfg, MujocoSimBackend
from sim.backends.mujoco.robot import MujocoRobot

__all__ = ["MujocoSimBackend", "MujocoBackendCfg", "MujocoRobot"]
