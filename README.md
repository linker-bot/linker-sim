# Workstation simulation

[English](README.md) · [中文](README.zh.md)

Bimanual RL simulation workspace for AR5 / P7 / A7-lite arms paired
with Linkerhand L6 / O6 / L25 / L30, running in a MuJoCo simulation
environment with real-robot telemetry replay.

<table>
  <tr>
    <td><img src="./docs/images/o6_urdf.png" alt="o6 URDF" width="250"/></td>
    <td><img src="./docs/images/dc_replay.gif" alt="Second Image" width="400"/></td>
  </tr>
</table>

## What this repo contains

- Composer-driven workstation assets (recipe → URDF + MJCF + manifest).
- A runtime backbone (`scripts/run.py`) that runs any composed
  workstation under MuJoCo with selectable controllers, tasks,
  and recorders.
- A real-robot telemetry replayer (`scripts/replay.py`).
- Validators, registry tools, and a CI drift gate for the asset
  pipeline.

## Project layout

This repo is a `uv` workspace with two members under `packages/`:

- `packages/linker-robot-assets/` — asset bundles + composer + validators.
  - `src/linker_robot_assets/assets/components/{arms,bases,hands}/` —
    reusable component subtrees (URDF + MJCF + meshes + `meta.yaml`).
  - `src/linker_robot_assets/assets/workstations/<name>/` — `recipe.yaml`
    plus generated `workstation.{urdf,mjcf}` and `manifest.yaml`.
  - `src/linker_robot_assets/composer/` — recipe → URDF/MJCF/manifest.
  - `src/linker_robot_assets/ci/check_drift.sh` — composer drift gate.
- `packages/linker-sim/` — sim runtime (depends on `linker-robot-assets`).
  - `src/linker_sim/backends/mujoco/` — MuJoCo backend implementation.
  - `src/linker_sim/controllers/` — `joint_pd`, `osc`, `ik`.
  - `src/linker_sim/tasks/` — task definitions.
  - `src/linker_sim/configs/` — Hydra configs (`pkg://linker_sim.configs`).

Top-level entry points and supporting trees:

- `scripts/run.py` / `scripts/replay.py` — Hydra entrypoints.
- `tests/` — pytest suite with synthetic fixtures.
- `docs/` — architecture, installation, usage, asset and MJCF authoring guides.

## Installation

See [docs/installation.md](docs/installation.md). MuJoCo-only
(Python 3.11 or 3.12, no GPU needed) — for replay, RL, and data
collection workflows.

Quick setup:

```bash
# robot assets live in the linker-sim-assets submodule (git-LFS meshes)
git submodule update --init
git -C packages/linker-robot-assets lfs pull   # requires git-lfs installed

python3 -m venv .venv-mujoco && source .venv-mujoco/bin/activate
pip install -e packages/linker-robot-assets -e packages/linker-sim[mujoco]
```

> **Submodule + LFS.** `packages/linker-robot-assets` is a git submodule
> ([linker-sim-assets](https://github.com/linker-bot/linker-sim-assets)),
> pinned to a release commit. Clone with `--recurse-submodules` (or run the
> two commands above), and `git lfs pull` inside it so the meshes materialize.

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

### Browser-based replay visualization

Browser-based replay visualization (WebGL, no GPU required) now lives
in the separate [`linker-sim-viser`](https://gitea.linkerhub.work/LinkerOS/linker-sim-viser)
repository, which consumes this repo's `linker-robot-assets` package as
its single source of truth for robot assets.

See [docs/known_limitations.md](docs/known_limitations.md) for the
hand-decoder linear-fit caveat and the UMI-Dex path-hack TODO.

## Development notes

- Keep generated artifacts out of git (`__pycache__`, virtual envs,
  logs); composed workstation files (`workstation.urdf`,
  `workstation.mjcf`, `manifest.yaml`) **are** committed.
- Put feature work on dedicated branches and commit before switching.
