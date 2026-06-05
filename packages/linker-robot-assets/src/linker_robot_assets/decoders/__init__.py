"""Decoders for telemetry channels → URDF joint angles.

Lives next to the asset bundles so consumers (UMI bag conversion,
future SDK clients, viser teleop) share one canonical mapping per
component. Today only `decode_hand` is exposed; expand here when arms
or other actuated subsystems gain SDK-shaped telemetry.

See `docs/known_limitations.md` for the linear-fit caveat.
"""

from linker_robot_assets.decoders.hand import CONVENTION, decode_hand

__all__ = ["CONVENTION", "decode_hand"]
