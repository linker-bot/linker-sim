"""Sim-agnostic controllers. See `base.py` for the Protocol.

Concrete controllers are not imported at package load time — they pull
in Isaac (OSC) or other backend-specific deps. Import them directly
when needed:

    from linker_sim.controllers.osc import OscController, OscControllerCfg
    from linker_sim.controllers.joint_pd import JointPDController, JointPDControllerCfg
    from linker_sim.controllers.ik import IkController, IkControllerCfg
"""

from linker_sim.controllers.base import Controller

__all__ = ["Controller"]
