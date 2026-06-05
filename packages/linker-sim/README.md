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

From the repository root. `uv` honours the workspace's `[tool.uv.sources]`
and resolves `linker-robot-assets` automatically; plain `pip` does not, so
install both workspace members explicitly.

```bash
# uv (single command resolves linker-robot-assets from the workspace)
uv pip install -e 'packages/linker-sim[mujoco]'

# plain pip — install both members
pip install -e packages/linker-robot-assets -e packages/linker-sim[mujoco]
pip install -e packages/linker-robot-assets -e packages/linker-sim[isaac]   # in env_isaaclab
pip install -e packages/linker-robot-assets -e packages/linker-sim[tools]   # composer / validator only
```
