# linker-sim

Sim runtime + RL backbone for the LinkerOS workstations: Hydra entrypoints,
Isaac Sim and MuJoCo backends, controllers, tasks, replay, recorders. Asset
data and the composer ship in a sibling package, [`linker-robot-assets`](../linker-robot-assets/),
which this package depends on.

This `pyproject.toml` is a workspace member of the repository at the parent
directory. See the [repository README](../../README.md) and
[docs/installation.md](../../docs/installation.md) for the full setup,
extras (`[mujoco]`, `[isaac]`, `[tools]`, `[lerobot]`, `[dev]`, `[all]`),
and usage.

## Install

From the repository root:

```bash
pip install -e packages/linker-sim[mujoco]      # MuJoCo-only profile
pip install -e packages/linker-sim[isaac]       # Isaac Sim profile (in env_isaaclab)
pip install -e packages/linker-sim[tools]       # composer / validator only
```

`linker-robot-assets` is resolved from the workspace via `[tool.uv.sources]`
when installed under `uv`; under plain `pip`, install both packages
explicitly.
