# Installation Guide

This repo runs entirely on MuJoCo — no GPU, Isaac Sim, or Isaac Lab
required. Python 3.11 or 3.12 on Linux.

---

## Install

Use this for the MuJoCo backend — replaying telemetry on a
data-collection workstation (a7_lite), MJCF validation, RL rollouts, and
composing/validating workstations.

### Prerequisites

- Ubuntu 22.04 or 24.04
- Python 3.11 or 3.12
- `git`
- `git-lfs` (robot-asset meshes are LFS-tracked in the submodule)
- `uv` (or `pip`)

### Setup

```bash
cd /path/to/linker-sim

# Robot assets live in the linker-sim-assets submodule (git-LFS meshes),
# pinned to a release commit. Initialize it and pull the meshes.
git submodule update --init
git -C packages/linker-robot-assets lfs pull

# Create a venv with your system Python (3.11 or 3.12)
python3 -m venv .venv-mujoco
source .venv-mujoco/bin/activate

# Install the MuJoCo subset (pulls mujoco, torch, pyyaml, hydra-core).
# The repo root is a uv workspace; extras live on the linker-sim package.
# Plain pip doesn't honor [tool.uv.sources], so install both workspace
# members explicitly.
pip install -e packages/linker-robot-assets -e packages/linker-sim[mujoco]
```

> **Source-checkout only.** Use editable installs (`pip install -e`).
> The composer assets, Hydra configs, and `scripts/` entrypoints are
> resolved from the source tree, not from package data. Building and
> distributing a wheel is **not** a supported workflow.

Optional extras (attach to `packages/linker-sim`):

- `[tools]` — composer + validator + registry (CPU-safe, no MuJoCo)
- `[mujoco]` — the above plus `mujoco` + `torch` (runtime + MJCF authoring)
- `[lerobot]` — the above plus `pyarrow` (LeRobot parquet recorder sink)
- `[dev]` — `ruff`, `pytest`
- `[all]` — `tools` + `mujoco` + `lerobot`

### Verify — replay smoke test

```bash
python scripts/replay.py robot=a7_lite_l6_dc source=data_collection headless=true max_frames=50
```

### Verify — rollout smoke test

```bash
python scripts/run.py max_steps=200 headless=true
```

Explicit workstation selection:

```bash
python scripts/run.py robot=p7_i1_o6_bimanual max_steps=200 headless=true
```

> The default workstation is `ar5_o6_bench_bimanual` (AR5 arms + Linker
> O6 hands). Other shipped workstations: `ar5_08_o6_bench_bimanual`,
> `p7_i1_o6_bimanual`, `a7_lite_o6_dc`; `ar5_l25_bench_bimanual`,
> `ar5_08_l25_bench_bimanual`, `p7_i1_l25_bimanual`, `a7_lite_l25_dc`;
> plus the L6-hand variants (`ar5_l6_bench_bimanual`,
> `p7_i1_l6_bimanual`, `a7_lite_l6_dc`) for backwards compatibility;
> and the L20 Lite variants (`p7_i1_l20lite_bimanual`,
> `a7_lite_l20lite_dc`).

See [USAGE.md](USAGE.md) for the full set of `scripts/run.py` and
`scripts/replay.py` knobs.

### Compose and validate workstations

The composer and validator need only `packages/linker-sim[tools]` (and
the `linker-robot-assets` member it depends on) — `mujoco` is used for
the MJCF-parity checks.

```bash
# Recompose one workstation after editing its recipe / a referenced component
python -m linker_robot_assets.composer.compose packages/linker-robot-assets/src/linker_robot_assets/assets/workstations/ar5_l6_bench_bimanual

# Recompose everything
for ws in packages/linker-robot-assets/src/linker_robot_assets/assets/workstations/*/; do python -m linker_robot_assets.composer.compose "$ws"; done

# Validate (manifest hashes, URDF kinematics, mesh resolution, composer drift, MJCF parity)
python -m linker_robot_assets.validate_workstation packages/linker-robot-assets/src/linker_robot_assets/assets/workstations/ar5_l6_bench_bimanual

# List composed workstations
python -m linker_sim.tools.registry_show

# Dump the registry handle for one workstation
python -m linker_sim.tools.registry_show ar5_l6_bench_bimanual

# CI drift check (fails if committed artifacts are stale)
bash packages/linker-robot-assets/src/linker_robot_assets/ci/check_drift.sh
```

### Daily activation

```bash
source /path/to/linker-sim/.venv-mujoco/bin/activate
cd /path/to/linker-sim
```

## Troubleshooting

- **`ModuleNotFoundError: linker_sim` or `linker_robot_assets`** — You
  haven't run `pip install -e packages/linker-robot-assets -e packages/linker-sim[mujoco]`
  in the active env, or the wrong env is active.
- **Wrong Python interpreter in shell** — `which python` and re-activate
  the intended venv.
- **Asset path errors** — Composed workstation URDFs use relative paths
  like `../../components/…/meshes/…`. Run commands from this repo's root
  so paths resolve correctly, or pass absolute paths.
