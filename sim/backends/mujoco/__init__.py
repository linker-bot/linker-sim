"""MuJoCo backend — stub only.

The full MuJoCo implementation depends on `workstation.mjcf` artifacts,
which don't exist until PR #1b lands (component MJCFs are hand-authored
per `docs/component_mjcf_authoring.md`). This package exists so the
Protocol in `sim/backends/base.py` has a second declared target —
designing for one backend invites Isaac-shaped assumptions. Every entry
point raises `NotImplementedError` with the exact unblock signal.

When PR #1b lands, the stubs become:
- `backend.py` — wraps `mujoco.MjModel` + `mujoco.MjData`, B=1 per
  process (D8), numpy↔torch conversion at the boundary (D10).
- `robot.py` — mass matrix via `mj_fullM`, Jacobian via `mj_jac`,
  gravity via `mj_rnePostConstraint`.
"""

from __future__ import annotations

from sim.backends.mujoco.backend import MujocoSimBackend, MujocoBackendCfg
from sim.backends.mujoco.robot import MujocoRobot

__all__ = ["MujocoSimBackend", "MujocoBackendCfg", "MujocoRobot"]
