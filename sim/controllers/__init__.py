"""Sim-agnostic controllers. See `base.py` for the Protocol.

Concrete controllers are not imported at package load time — they pull
in Isaac (OSC) or other backend-specific deps. Import them directly
when needed:

    from sim.controllers.osc import OscController, OscControllerCfg
    from sim.controllers.joint_pd import JointPDController, JointPDControllerCfg
    from sim.controllers.ik import IkController, IkControllerCfg
"""

from sim.controllers.base import Controller

__all__ = ["Controller"]
