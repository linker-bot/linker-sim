# Installation Guide

This project assumes Linux with a working NVIDIA GPU stack. IsaacLab is installed separately, then used to run this repository.

## Prerequisites

- Ubuntu 22.04 or 24.04 (both officially supported by Isaac Sim 5.x)
- NVIDIA GPU with RT Cores (RTX 20-series or newer)
- NVIDIA driver >= 580.65.06 (tested baseline for Isaac Sim 5.x)
- `git`
- `uv` package manager
- **Python 3.11 exactly** (required by Isaac Sim 5.x; 3.10 and 3.12 will not work)

On Ubuntu 24.04 the system default is Python 3.12, and Python 3.11 is not available from the default apt repos. Use one of:

- **Recommended — let `uv` fetch a standalone 3.11 build (no apt / PPA needed):**

  ```bash
  uv python install 3.11
  ```

- **Alternative — install from the deadsnakes PPA:**

  ```bash
  sudo add-apt-repository ppa:deadsnakes/ppa
  sudo apt update
  sudo apt install python3.11 python3.11-venv
  ```

Install `uv` (if needed):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 1) Create and activate local venv (uv)

From repo root, pin the venv to Python 3.11:

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
python --version  # should report 3.11.x
```

## 2) Install repo-level Python requirements

```bash
uv pip install -r requirements.txt
```

Note: this `requirements.txt` is only for this repo's extra dependencies. IsaacLab/Isaac Sim packages are managed in the IsaacLab setup step below.

## 3) Install IsaacLab under `docs/`

This repository does not vendor IsaacLab source. Clone IsaacLab locally into `docs/IsaacLab` and check out a released tag that matches Isaac Sim 5.x (pinning avoids surprises from `main`):

```bash
cd docs
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout v2.3.2
./isaaclab.sh --help
```

Recommended flow (uv environment managed by IsaacLab tooling):

```bash
# 3a) Create the IsaacLab-managed uv environment (Python 3.11)
./isaaclab.sh -u

# 3b) Activate it so subsequent `uv pip` calls target the right env
source env_isaaclab/bin/activate        # bash/zsh
# source env_isaaclab/bin/activate.fish # fish

# 3c) Install Isaac Sim 5.0 into the env BEFORE running `-i`.
# Isaac Lab's `-i` step installs the Lab extensions, but it assumes
# the `isaacsim` package is already present in the environment.
uv pip install 'isaacsim[all,extscache]==5.0.0' \
    --extra-index-url https://pypi.nvidia.com

# 3d) Preempt a known flatdict build failure (flatdict 4.0.1 does not
# declare pkg_resources as a build dep, which modern setuptools 80+
# no longer exposes implicitly). Install it first without build isolation.
uv pip install setuptools
uv pip install flatdict==4.0.1 --no-build-isolation

# 3e) Now install Isaac Lab extensions + RL extras
./isaaclab.sh -i
```

Key commands from IsaacLab script usage:

- `./isaaclab.sh -u`: create IsaacLab uv environment
- `./isaaclab.sh -i`: install IsaacLab extensions + RL extras (requires `isaacsim` already installed)
- `./isaaclab.sh -p <script.py ...>`: run python with IsaacLab runtime context

To run this repo's scripts you must have the IsaacLab env active (`source docs/IsaacLab/env_isaaclab/bin/activate[.fish]`), or invoke scripts via `./docs/IsaacLab/isaaclab.sh -p <script.py ...>`.

## 4) Run this repository OSC smoke test (single and multi-robot)

From repository root, run one of the OSC smoke-test modes:

```bash
python sim/envs/test_osc/spawn_osc_scene.py --num_envs 1 --robot_side left
```

Right-arm mode:

```bash
python sim/envs/test_osc/spawn_osc_scene.py --num_envs 1 --robot_side right
```

Dual-arm mode:

```bash
python sim/envs/test_osc/spawn_osc_scene.py --num_envs 1 --robot_side both
```

Multi-env mode: 

```bash
python sim/envs/test_osc/spawn_osc_scene.py --num_envs 16
```

## 5) Tune OSC gains (optional, recommended)

```bash
python sim/envs/test_osc/gain_tuner_osc.py --num_envs 1 --robot_side left
```

This creates `sim/envs/test_osc/osc_gains.json` on first run and hot-reloads updates while running.


## Troubleshooting

- `ModuleNotFoundError: isaaclab`
  - Ensure IsaacLab setup has been completed (`-u` and `-i`), and you are using the intended environment.
- `FileNotFoundError: Could not find the isaac-sim directory ... _isaac_sim` during `./isaaclab.sh -i`
  - The `isaacsim` pip package was not installed before `-i`. Install it first: `uv pip install 'isaacsim[all,extscache]==5.0.0' --extra-index-url https://pypi.nvidia.com`, then re-run `-i`.
- `Failed to build flatdict==4.0.1` / `ModuleNotFoundError: No module named 'pkg_resources'`
  - Install flatdict with build isolation disabled: `uv pip install setuptools && uv pip install flatdict==4.0.1 --no-build-isolation`, then re-run `-i`.
- Isaac Sim import errors about Python version / ABI mismatch
  - Confirm your venv Python is 3.11 (`python --version`). Isaac Sim 5.x will not load under 3.10 or 3.12.
- Isaac Sim window does not launch or crashes early
  - Verify NVIDIA driver/GPU compatibility and system graphics stack.
- Wrong Python interpreter in shell
  - Check `which python` and re-activate your virtual environment.
- Asset path errors
  - Run commands from repository root so relative paths resolve correctly.
