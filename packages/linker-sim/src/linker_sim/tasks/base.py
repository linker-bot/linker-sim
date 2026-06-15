"""Task Protocol.

A `Task` owns the env's obs / reward / done semantics and any
task-local reset state (target sampling, object spawning). `BaseEnv`
drives it through a narrow contract so swapping tasks (reach → pick →
place) is a config change, not a code change.

Design points:

- Tasks don't own the simulator. They ask the backend for state
  (`backend.robots[...]`) and for scene-level primitives when needed
  (`backend.spawn_rigid(...)`).
- Obs shape is `(B, observation_dim)`. `observation_dim` must be
  knowable at construction time (before any physics step) so `BaseEnv`
  can validate against the policy's expected input size.
- Action shape is `(B, action_dim)`. `action_dim` must equal the sum
  of attached controllers' `command_dim`s; `BaseEnv` enforces this.
- Termination and truncation are separate flags, gym-conventional:
  `terminated` = task condition hit (success/failure), `truncated` =
  time/step limit.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch

from linker_sim.backends.base import SimBackend


@runtime_checkable
class Task(Protocol):
    """Env-facing contract for a task definition."""

    observation_dim: int
    action_dim: int

    def reset(self, backend: SimBackend, env_ids: torch.Tensor) -> None:
        """Task-local reset for `env_ids` (target resampling, object
        pose redraw). Backend-level reset (root + joints) already ran."""
        ...

    def observe(
        self, backend: SimBackend, last_action: torch.Tensor
    ) -> torch.Tensor:
        """Return `(B, observation_dim)` observation tensor."""
        ...

    def reward(
        self, backend: SimBackend, last_action: torch.Tensor
    ) -> torch.Tensor:
        """Return `(B,)` per-env reward."""
        ...

    def done(
        self,
        backend: SimBackend,
        step_count: torch.Tensor,
        max_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `(terminated, truncated)`, each `(B,)` bool."""
        ...
