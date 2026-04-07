# dex-tool-rl

RL simulation workspace for AR5 + Linkerhand L6 robots using IsaacLab/Isaac Sim.

## What this repo contains

- A minimal smoke-test simulation scene for AR5_L6 robot assets.
- Scene and robot asset configuration helpers.
- Early-stage task spec notes for future RL task/reward design.

## Current status

This repository is in active setup/prototyping stage. The current runnable entrypoint is a smoke-test scene under `sim/envs/test`. The task spec in `docs/target_spec.md` is a draft template and not fully filled.

## Project layout

- `sim/assets`: robot and scene asset configs (URDF-backed `ArticulationCfg`/`AssetBaseCfg`)
- `sim/envs/test`: smoke-test scene configs and runner script
- `docs/target_spec.md`: deterministic interface template for RL task definition
- `docs/installation.md`: setup guide (including local IsaacLab clone workflow)

## Installation

Use the setup guide in `docs/installation.md` for:

- `uv` + `venv` environment setup
- dependency installation via `requirements.txt`
- IsaacLab clone/setup and integration
- first-run troubleshooting

## Quick start

After completing `docs/installation.md`, run the smoke-test scene:

```bash
python sim/envs/test/spawn_scene.py --num_envs 1 --robot_side left
```

Optional dual-arm smoke test:

```bash
python sim/envs/test/spawn_scene.py --num_envs 1 --robot_side both
```

Expected behavior:

- Isaac Sim app launches.
- Scene spawns robot(s) and workstation table.
- Console prints setup/reset logs such as `[INFO] Test scene setup complete.`

## Development notes

- Keep generated artifacts out of git (`__pycache__`, virtual envs, logs).
- Put feature work on dedicated branches and commit before switching branches.
- Keep local third-party checkouts (for example `docs/IsaacLab`) out of this repo history.
