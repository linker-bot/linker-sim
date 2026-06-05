# Workstation simulation

Bimanual RL simulation workspace for AR5 + Linkerhand L6 / LKLS73 / a7_lite
robots, with both Isaac Sim and MuJoCo backends.

## What this repo contains

- Composer-driven workstation assets (recipe → URDF + MJCF + manifest).
- A runtime backbone (`scripts/run.py`) that runs any composed
  workstation under either backend with selectable controllers, tasks,
  and recorders.
- A real-robot telemetry replayer (`scripts/replay.py`).
- Validators, registry tools, and a CI drift gate for the asset
  pipeline.

## Project layout

- `assets/components/{arms,bases,hands}/` — reusable component
  subtrees (URDF + MJCF + meshes + `meta.yaml`).
- `assets/workstations/<name>/` — `recipe.yaml` plus generated
  `workstation.{urdf,mjcf}` and `manifest.yaml`.
- `sim/backends/{isaac,mujoco}/` — backend implementations.
- `sim/controllers/` — `joint_pd`, `osc`, `ik`.
- `sim/tasks/` — task definitions.
- `sim/envs/test_osc/` — interactive OSC gain tuner.
- `scripts/run.py` / `scripts/replay.py` — Hydra entrypoints.
- `tools/` — composer, validators, registry inspector, drift gate.
- `docs/` — installation, usage, asset and MJCF authoring guides,
  test pipeline.

## Installation

See [docs/installation.md](docs/installation.md). Two profiles are available:

- **MuJoCo-only** (Python 3.11 or 3.12, no GPU needed) — for replay and data collection workflows.
- **Full** (Python 3.11 + NVIDIA GPU) — for Isaac Sim RL training.

Quick MuJoCo-only setup:

```bash
python3 -m venv .venv-mujoco && source .venv-mujoco/bin/activate
pip install -e packages/linker-sim[mujoco]
```

> **Source-checkout only.** Use editable installs (`pip install -e`).
> The composer assets, Hydra configs, and `scripts/` entrypoints are
> resolved from the source tree, not from package data. Building and
> distributing a wheel is **not** a supported workflow.

## Quick start

After installation, the smoke test:

```bash
python scripts/run.py max_steps=200 headless=true
```

For everything else (MuJoCo, replay, gain tuning, composing new
workstations, recording episodes), see [docs/USAGE.md](docs/USAGE.md)
([中文](docs/USAGE.zh.md)).

## Development notes

- Keep generated artifacts out of git (`__pycache__`, virtual envs,
  logs); composed workstation files (`workstation.urdf`,
  `workstation.mjcf`, `manifest.yaml`) **are** committed.
- Put feature work on dedicated branches and commit before switching.
- IsaacLab lives outside this repo (e.g. `~/opt/IsaacLab/`) and is
  shared across projects; see [docs/installation.md](docs/installation.md).
