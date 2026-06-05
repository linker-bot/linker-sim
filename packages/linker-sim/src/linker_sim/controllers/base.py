"""Controller Protocol.

Every controller in the repo conforms to this so `BaseEnv` can drive
them uniformly. The contract is thin on purpose: a controller owns
its own gain profile (applied to the robot once on `attach`) and its
own action→effort/target transformation.

Decimation model (understood by `BaseEnv`):

    env.step(action)
      for each controller:
        controller.set_command(action_slice, robot)
      for _ in range(decimation):
        for each controller:
          controller.apply(robot)
        backend.step()

`set_command` is called once per env step; `apply` is called
`decimation` times. Holds-over between calls is the controller's job
(implicit zero-order hold is a reasonable default).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch

from linker_sim.backends.base import Robot


@runtime_checkable
class Controller(Protocol):
    """Sim-agnostic controller writing to a single role on a `Robot`."""

    role: str
    """Role name this controller drives (must be a key in `robot.handle.joints`)."""

    command_dim: int
    """Width of the per-env command vector passed to `set_command`."""

    def attach(self, robot: Robot) -> None:
        """One-time setup: override gains, size internal buffers. Called
        by `BaseEnv` after the backend is constructed but before the
        first `reset`."""
        ...

    def set_command(self, command: torch.Tensor, robot: Robot) -> None:
        """Update the controller's target from the policy action (shape
        `(B, command_dim)`). May read current robot state."""
        ...

    def apply(self, robot: Robot) -> None:
        """Compute and write efforts or position targets to `robot`.
        Called every physics step during the decimation window."""
        ...

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Clear internal state for `env_ids`. Called by `BaseEnv` on
        env reset."""
        ...
