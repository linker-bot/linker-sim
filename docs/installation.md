# Installation Guide

This project supports two installation profiles:

| Profile | Python | GPU required | Use case |
|---------|--------|--------------|----------|
| **MuJoCo-only** | 3.11 or 3.12 | No (CPU-rendered viewer) | Replay, MJCF validation, data collection |
| **Full (Isaac + MuJoCo)** | 3.11 only | Yes (NVIDIA RTX) | RL training, Isaac Sim workflows |

Pick the section that matches your needs.

---

## A) MuJoCo-only install (Python 3.11 or 3.12)

Use this if you only need the MuJoCo backend — e.g. replaying telemetry on a data-collection workstation (a7_lite). No GPU or Isaac Sim required.

### Prerequisites

- Ubuntu 22.04 or 24.04
- Python 3.11 or 3.12
- `git`
- `uv` (or `pip`)

### Setup

```bash
cd /path/to/linker-sim

# Create a venv with your system Python (3.11 or 3.12)
python3 -m venv .venv-mujoco
source .venv-mujoco/bin/activate

# Install MuJoCo subset (pulls mujoco, pyyaml, hydra-core)
# The repo root is a uv workspace; extras live on the linker-sim package.
# Plain pip doesn't honor [tool.uv.sources], so install both workspace
# members explicitly.
pip install -e packages/linker-robot-assets -e packages/linker-sim[mujoco]
```

### Verify

```bash
python scripts/replay.py robot=a7_lite_dc source=data_collection headless=true max_frames=50
```

### Daily activation

```bash
source /path/to/linker-sim/.venv-mujoco/bin/activate
```

---

## B) Full install (Isaac Sim + MuJoCo, Python 3.11 only)

This project assumes Linux with a working NVIDIA GPU stack. IsaacLab is installed **outside this repo** at a shared location; this repo's two workspace members (`packages/linker-sim/` and `packages/linker-robot-assets/`) install on top via `pip` or `uv pip`.

One Python environment is used end-to-end: the IsaacLab-managed `env_isaaclab`. Composer, validator, registry tools, and runtime all share it.

### Prerequisites

- Ubuntu 22.04 or 24.04 (both officially supported by Isaac Sim 5.x)
- NVIDIA GPU with RT Cores (RTX 20-series or newer)
- NVIDIA driver >= 580.65.06 (tested baseline for Isaac Sim 5.x)
- `git`
- `uv` package manager
- **Python 3.11 exactly** (required by Isaac Sim 5.x; 3.12 will not work for this profile)

### Python 3.11 on Ubuntu 24.04

The system default is Python 3.12, and 3.11 is not available from the default apt repos. Use one of:

- **Recommended — let `uv` fetch a standalone 3.11 build:**
  ```bash
  uv python install 3.11
  ```
- **Alternative — deadsnakes PPA:**
  ```bash
  sudo add-apt-repository ppa:deadsnakes/ppa
  sudo apt update
  sudo apt install python3.11 python3.11-venv
  ```

### Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 1) Clone IsaacLab (outside this repo)

IsaacLab is a shared system dependency, not a vendored part of this repo. Put it at a canonical location such as `~/opt/IsaacLab/` so multiple projects can share one install.

```bash
mkdir -p ~/opt
cd ~/opt
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout v2.3.2     # pin to a tag matching Isaac Sim 5.1 (see §2c below)
```

> **Do not clone IsaacLab into this repo.** Earlier versions of this guide put it under `docs/IsaacLab/`; that convention is retired. `docs/` is for documentation only.

### 2) Create the IsaacLab venv + install Isaac Sim + Lab extensions

Run these from `~/opt/IsaacLab/` (wherever you cloned it):

```bash
# 2a) Create the IsaacLab uv env (Python 3.11). Creates ./env_isaaclab/
./isaaclab.sh -u

# 2b) Activate it. All subsequent installs target this env.
source env_isaaclab/bin/activate        # bash/zsh
# source env_isaaclab/bin/activate.fish # fish

# 2c) Install Isaac Sim 5.1 BEFORE running `-i`.
# Isaac Lab v2.3.2 requires isaacsim 5.1 (it pins isaacsim.asset.importer.urdf==2.4.31,
# which ships with 5.1 — not 5.0). Using 5.0 causes `./isaaclab.sh -i` to succeed
# but the first run of any script to fail with:
#   [isaaclab.python-2.3.2] dependency: 'isaacsim.asset.importer.urdf' = { version='=2.4.31' } can't be satisfied
uv pip install 'isaacsim[all,extscache]==5.1.0.0' \
    --extra-index-url https://pypi.nvidia.com

# 2d) Preempt a known flatdict build failure (flatdict 4.0.1 does not
# declare pkg_resources as a build dep, which modern setuptools 80+
# no longer exposes implicitly).
uv pip install setuptools
uv pip install flatdict==4.0.1 --no-build-isolation

# 2e) Now install Isaac Lab extensions + RL extras
./isaaclab.sh -i
```

IsaacLab scripts of note:

- `./isaaclab.sh -u`: create IsaacLab uv env (`env_isaaclab/`)
- `./isaaclab.sh -i`: install IsaacLab extensions (requires `isaacsim` already present in the active env)
- `./isaaclab.sh -p <script.py …>`: run Python with IsaacLab runtime context

### 3) Install this repo's packages into the same env

From this repo's root, with `env_isaaclab` still activated:

```bash
cd /path/to/linker-sim
uv pip install -e 'packages/linker-sim[tools]'          # composer / validator / registry
# Or: `packages/linker-sim[all]` once you want MJCF tooling (pulls mujoco).
```

The repo root is a `uv` workspace with no `[project]` table; extras live
on the `linker-sim` member, so install paths reference `packages/linker-sim`
rather than `.`. This installs `linker_sim/` (and resolves
`linker-robot-assets` from the workspace) into the active env
(`env_isaaclab`), pulls `pyyaml` for the composer, and registers the
project for editable imports so `from linker_sim.registry import load`
works without `sys.path` hacks.

Optional extras (attach to `packages/linker-sim`):

- `[tools]` — composer + validator + registry (CPU-safe; no Isaac)
- `[mujoco]` — the above plus `mujoco` (for MJCF authoring/validation)
- `[isaac]` — Isaac Sim + flatdict (only needed if you're *not* installing IsaacLab's Isaac Sim from step 2c; most users skip this)
- `[dev]` — `ruff`, `pytest`
- `[all]` — `tools` + `mujoco` + `isaac`

For daily work with IsaacLab already installed,
`uv pip install -e 'packages/linker-sim[tools]'` is usually enough.

### 4) Verify — Isaac smoke test

With `env_isaaclab` activated:

```bash
cd /path/to/linker-sim
python scripts/run.py max_steps=200 headless=true
```

Explicit workstation selection:

```bash
python scripts/run.py robot=lkls73_i1_o6_bimanual max_steps=200 headless=true
```

> The default workstation is `ar5_o6_bench_bimanual` (AR5 arms + Linker
> O6 hands). Other shipped workstations: `lkls73_i1_o6_bimanual`,
> `a7_lite_o6_dc`, plus the L6-hand variants (`ar5_l6_bench_bimanual`,
> `lkls73_i1_bimanual`, `a7_lite_dc`) for backwards compatibility.

Multi-env:

```bash
python scripts/run.py num_envs=16 max_steps=200 headless=true
```

See [USAGE.md](USAGE.md) for the full set of `scripts/run.py` knobs and
the MuJoCo backend.

### 5) Compose and validate workstations

The composer and validator don't need Isaac Sim — just `packages/linker-sim[tools]` (and the `linker-robot-assets` member it depends on).

```bash
# Recompose one workstation after editing its recipe / a referenced component
python -m linker_robot_assets.composer.compose packages/linker-robot-assets/src/linker_robot_assets/assets/workstations/ar5_l6_bench_bimanual

# Recompose everything
for ws in packages/linker-robot-assets/src/linker_robot_assets/assets/workstations/*/; do python -m linker_robot_assets.composer.compose "$ws"; done

# Validate (8 checks: manifest hashes, kinematic structure, mesh resolution, composer drift)
python -m linker_robot_assets.validate_workstation packages/linker-robot-assets/src/linker_robot_assets/assets/workstations/ar5_l6_bench_bimanual

# List composed workstations
python -m linker_sim.tools.registry_show

# Dump the registry handle for one workstation
python -m linker_sim.tools.registry_show ar5_l6_bench_bimanual

# CI drift check (fails if committed artifacts are stale)
bash packages/linker-robot-assets/src/linker_robot_assets/ci/check_drift.sh
```

### Daily activation (full install)

After setup, a typical day looks like:

```bash
source ~/opt/IsaacLab/env_isaaclab/bin/activate
cd /path/to/linker-sim
# ... work ...
```

Alias it if you want:

```bash
alias dexrl='source ~/opt/IsaacLab/env_isaaclab/bin/activate && cd /path/to/linker-sim'
```

## Troubleshooting

- **`ModuleNotFoundError: isaaclab`** — Ensure IsaacLab setup (`-u` and `-i`) completed and `env_isaaclab` is active. `which python` should point inside `env_isaaclab/bin/`.
- **`FileNotFoundError: Could not find the isaac-sim directory … _isaac_sim`** during `./isaaclab.sh -i` — `isaacsim` wasn't installed before `-i`. Install it first: `uv pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com`, then re-run `-i`.
- **`[isaaclab.python-…] dependency: 'isaacsim.asset.importer.urdf' = { version='=2.4.31' } can't be satisfied`** at runtime — Isaac Sim version mismatch. IsaacLab v2.3.2 needs Isaac Sim 5.1; if you installed 5.0, upgrade: `uv pip uninstall isaacsim && uv pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com && ./isaaclab.sh -i`.
- **`Failed to build flatdict==4.0.1` / `ModuleNotFoundError: No module named 'pkg_resources'`** — Install flatdict with build isolation disabled: `uv pip install setuptools && uv pip install flatdict==4.0.1 --no-build-isolation`, then re-run `-i`.
- **Isaac Sim import errors about Python version / ABI mismatch** — Confirm your venv Python is 3.11 (`python --version`). Isaac Sim 5.x does not load under 3.10 or 3.12. The MuJoCo-only profile works fine on 3.12.
- **`ModuleNotFoundError: linker_sim` or `linker_robot_assets`** — You haven't run `uv pip install -e 'packages/linker-sim[tools]'` in the active env (which transitively pulls `linker-robot-assets` from the workspace), or the wrong env is active.
- **Isaac Sim window doesn't launch / crashes early** — Verify NVIDIA driver/GPU compatibility and system graphics stack.
- **Wrong Python interpreter in shell** — `which python` and re-activate the intended venv.
- **Asset path errors** — Composed workstation URDFs use relative paths like `../../components/…/meshes/…`. Run commands from this repo's root so paths resolve correctly, or pass absolute paths.

## Migrating from the old layout

If you previously had IsaacLab cloned under `docs/IsaacLab/` plus a repo-root `.venv/`:

```bash
# 1. Remove the old repo-local venv (no longer used)
rm -rf .venv

# 2. Move IsaacLab clone out of the repo (or re-clone outside)
mv docs/IsaacLab ~/opt/IsaacLab
# ... env_isaaclab paths inside env_isaaclab/bin/activate may hold the old
# prefix. Simplest fix is to delete env_isaaclab and re-run `./isaaclab.sh -u`.

# 3. Proceed with step 2 above, then 3 (install this repo's packages).
```
