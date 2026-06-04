"""OSC (operational-space control) controller — NOT IMPLEMENTED.

TODO: Rewrite and test before use. The previous implementation was never
validated end-to-end and has been gutted.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from linker_sim.backends.base import Robot


@dataclass
class OscControllerCfg:
    role: str = "arm"
    frame: str | None = None

    action_scale_pos: float = 0.05
    action_scale_rot: float = 0.25

    stiffness: tuple[float, ...] = (150.0, 150.0, 150.0, 80.0, 80.0, 80.0)
    damping_ratio: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    nullspace_stiffness: float = 10.0
    nullspace_damping_ratio: float = 1.0

    actuator_stiffness: float = 150.0
    actuator_damping: float = 8.0
    gain_profile: str | None = "osc"


class OscController:
    """Isaac-backed OSC controller — stub, not implemented."""

    role: str
    command_dim: int = 6

    def __init__(self, cfg: OscControllerCfg | None = None):
        self.cfg = cfg or OscControllerCfg()
        self.role = self.cfg.role

    def attach(self, robot: Robot) -> None:
        # TODO: implement OSC attach
        raise NotImplementedError("OscController is not implemented")

    def set_command(self, command: torch.Tensor, robot: Robot) -> None:
        # TODO: implement OSC set_command
        raise NotImplementedError("OscController is not implemented")

    def apply(self, robot: Robot) -> None:
        # TODO: implement OSC apply
        raise NotImplementedError("OscController is not implemented")

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        # TODO: implement OSC reset
        raise NotImplementedError("OscController is not implemented")
