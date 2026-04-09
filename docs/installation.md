# Installation Guide

This project assumes Linux with a working NVIDIA GPU stack. IsaacLab is installed separately, then used to run this repository.

## Prerequisites

- Ubuntu/Linux host
- NVIDIA driver compatible with your Isaac Sim version
- `git`
- `uv` package manager
- Python 3.10 or 3.11 (match your IsaacLab/Isaac Sim setup)

Install `uv` (if needed):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 1) Create and activate local venv (uv)

From repo root:

```bash
uv venv .venv
source .venv/bin/activate
python --version
```

## 2) Install repo-level Python requirements

```bash
uv pip install -r requirements.txt
```

Note: this `requirements.txt` is only for this repo's extra dependencies. IsaacLab/Isaac Sim packages are managed in the IsaacLab setup step below.

## 3) Install IsaacLab under `docs/`

This repository does not vendor IsaacLab source. Clone IsaacLab locally into `docs/IsaacLab`:

```bash
cd docs
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --help
```

Recommended flow (uv environment managed by IsaacLab tooling):

```bash
./isaaclab.sh -u
./isaaclab.sh -i
```

Key commands from IsaacLab script usage:

- `./isaaclab.sh -u`: create IsaacLab uv environment
- `./isaaclab.sh -i`: install IsaacLab extensions + RL extras
- `./isaaclab.sh -p <script.py ...>`: run python with IsaacLab runtime context

## 4) Run this repository smoke test (single and multi-robot)

From repository root, run one of the smoke-test modes:

```bash
python sim/envs/test/spawn_scene.py --num_envs 1 --robot_side left
```

Right-arm mode:

```bash
python sim/envs/test/spawn_scene.py --num_envs 1 --robot_side right
```

Dual-arm mode:

```bash
python sim/envs/test/spawn_scene.py --num_envs 1 --robot_side both --reset_interval 600
```

Multi-env mode: 

```bash
python sim/envs/test/spawn_scene.py --num_envs 16 --reset_interval 120 --reset_envs_per_event 4
```


## Troubleshooting

- `ModuleNotFoundError: isaaclab`
  - Ensure IsaacLab setup has been completed (`-u` and `-i`), and you are using the intended environment.
- Isaac Sim window does not launch or crashes early
  - Verify NVIDIA driver/GPU compatibility and system graphics stack.
- Wrong Python interpreter in shell
  - Check `which python` and re-activate your virtual environment.
- Asset path errors
  - Run commands from repository root so relative paths resolve correctly.
