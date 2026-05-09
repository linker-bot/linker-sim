"""Task definitions. Canonical location for the Task Protocol + concrete
task classes used by `sim.envs.base.BaseEnv`.

Concrete tasks aren't eagerly imported — some pull Isaac (object
spawning) and we want `sim.tasks.base` to stay lightweight for
non-Isaac contexts.
"""

from sim.tasks.base import Task

__all__ = ["Task"]
