# Known limitations

Caveats that are intentional, tracked, and shipped behind a flag /
placeholder rather than fixed today. Each entry names the workaround
and the trigger for revisiting it.

## Hand decoder linear fit

[`linker_robot_assets.decoders.decode_hand`](../packages/linker-robot-assets/src/linker_robot_assets/decoders/hand.py)
maps an SDK 0–100 percent value per channel to URDF `[lower, upper]`
limits using a plain linear interpolation. The Linker Hand SDK has not
yet defined an angle convention, so this is a placeholder: the decoder
exports `CONVENTION = "linear-fit-v0"` and stamps that string into
written outputs (see the
[scripts/umi_bag_to_ee_poses.py](../scripts/umi_bag_to_ee_poses.py)
output payload).

What's wrong: the real per-finger response curve is almost certainly
non-linear, and may invert direction on some joints. The vendor curves
are not yet available.

What's right: the convention is consistent across the codebase. Today
both [`linker_sim.io.replay.hands`](../packages/linker-sim/src/linker_sim/io/replay/hands.py)
(byte-scale 0–255 for bag replay) and `linker_robot_assets.decoders.hand`
(SDK-scale 0–100 for UMI conversion) use the same direction, verified
empirically against Linker Hand O6 telemetry: `raw=full-scale → joint=
lower limit (rest / open)`, `raw=0 → joint=upper limit (full travel)`.

When to revisit: the Linker SDK ships an angle convention. At that
point:

- Bump `CONVENTION = "sdk-vN"` in
  `linker_robot_assets/decoders/hand.py`.
- Migrate or re-stamp any bagged data carrying `linear-fit-v0` (grep
  `decoder_convention` across stored `.npz` outputs).
- Unify the `linker_sim.io.replay.hands` byte decoders with the new
  convention (single shared decoder).

## UMI-Dex path hack

[`scripts/umi_bag_to_ee_poses.py`](../scripts/umi_bag_to_ee_poses.py)
and [`scripts/anchor_search.py`](../scripts/anchor_search.py) import
the [UMI-Dex](https://github.com/google-deepmind/umi-dex) Python API
via a `sys.path.insert(0, "~/codes/UMI-Dex/src")` hack at the top of
each script. This is fragile — it requires every consumer to have a
local UMI-Dex checkout at that exact path.

What's wrong: the path hack breaks any clone of `linker-sim` that
doesn't also have `~/codes/UMI-Dex`. Distributing the scripts as part
of a public release would surface this immediately.

What's right (for now): UMI-Dex is itself in active development and
hasn't published a stable Python API. Pinning a PyPI release would be
premature.

When to revisit: UMI-Dex publishes its `umi_dex` package to PyPI or an
internal index. At that point:

- Replace the `sys.path` hacks with `from umi_dex.controllers.calibrate
  import Calibrator`.
- Add `umi-dex>=X.Y` to a new `[umi-replay]` extra on
  `packages/linker-sim/pyproject.toml`.
- Document the install path under the data-collection-team section of
  the README.
