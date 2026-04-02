# Isaac Gym -> Isaac Lab Migration Memory

Last updated: 2026-04-02

This file stores the migration plan from the original `simtoolreal` setup (Isaac Gym) to this fresh repo `dex_tool_rl` (Isaac Lab). Keep this file in git so it is available on any PC after clone/pull.

## 1) Derivative-work status snapshot

- Current state: `dex_tool_rl` started as a fresh scaffold, not a direct copy.
- If substantial code/config is copied from `simtoolreal`, this project becomes derivative in practice.
- `simtoolreal` is MIT-licensed, so copying is allowed with attribution/license preservation.

## 2) Migration strategy (recommended)

- Prefer incremental porting:
  - first stand up Isaac Lab toolchain,
  - then port env/task behavior in small validated chunks.
- Keep behavior parity tests while migrating; do not assume API-level equivalence between Isaac Gym and Isaac Lab.

## 3) Toolchain setup on a machine

Old project used:
- Isaac Gym + Python 3.8 + custom `isaacgymenvs` + `rl_games`.

New project target:
- Isaac Lab + Isaac Sim pip package + Python 3.11.

### Basic setup commands

```bash
cd /path/to/dex_tool_rl
uv venv --python 3.11 .venv
source .venv/bin/activate
pip install --upgrade pip
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
```

Install PyTorch matching your CUDA stack (choose one compatible set from Isaac Lab docs), then:

```bash
git clone https://github.com/isaac-sim/IsaacLab.git /path/to/IsaacLab
cd /path/to/IsaacLab
./isaaclab.sh --install
```

Smoke test:

```bash
isaacsim
```

Accept EULA on first launch.

## 4) Project scaffold in this repo

Keep/organize:
- `sim/` for environment/task code.
- `configs/` for env and training configs.
- `rl/` for runner wrappers and eval scripts.
- `assets/` for URDF/USD/meshes.

If using template generation:
- Use Isaac Lab external project template (`isaaclab.sh --new`) and adapt folder layout to this repo.

## 5) API mapping guide (Gym -> Lab)

Source references from old repo:
- `simtoolreal/isaacgymenvs/tasks/simtoolreal/env.py`
- `simtoolreal/isaacgymenvs/cfg/task/SimToolReal.yaml`
- `simtoolreal/isaacgymenvs/launch_training.py`

Conceptual mapping:
- `VecTask` -> Isaac Lab RL env base (Direct workflow style is closest).
- `create_sim()` -> scene setup (`_setup_scene`).
- `pre_physics_step()` -> action processing/application.
- `post_physics_step()` -> obs/reward/done/update flow.
- `reset_idx()` -> `_reset_idx(env_ids)`.

## 6) Port order (strict)

1. Scene only: robot + table + object spawn.
2. Action path: joint targets and clipping.
3. Observation path: port obs/state terms incrementally.
4. Reset/termination: timeout, drop, success-steps.
5. Reward terms: one term at a time with scale parity.
6. Domain randomization: friction/mass/noise/delay last.

## 7) Known high-risk items from old config

- Action/observation delay and object-state delay/noise.
- Very large env counts (8k+).
- Custom launcher tuning and non-default RL settings.

Recommendation:
- Start with baseline PPO in Isaac Lab defaults.
- Add delay/noise and advanced trainer behavior only after stable baseline learning.

## 8) Parity checklist before long training

- Joint order, limits, and control mode match.
- Physics dt, action repeat/control frequency match.
- Reset distributions and object spawn logic match.
- Reward formula and scales match.
- Success metric (`successTolerance`, staged success steps) match.
- Failure conditions (drop threshold, timeout, collisions) match.

## 9) Bring-up sequence for stable migration

1. 1 env + rendering + debug values.
2. 64-256 envs headless.
3. scale env count while watching GPU memory/FPS.
4. compare learning curve against one canonical old task.

## 10) New-PC quick start checklist

When moving to another computer:

1. Clone repo and open this file:
   - `docs/isaac_lab_migration_memory.md`
2. Create Python 3.11 environment and install Isaac Sim/Isaac Lab deps.
3. Verify GPU driver/CUDA compatibility with chosen torch wheels.
4. Run `isaacsim` once (EULA + startup validation).
5. Run a tiny tutorial or sample Isaac Lab training script.
6. Resume migration from section 6 ("Port order").

## 11) Notes to future self

- Keep migration notes updated whenever architecture decisions change.
- If any code is copied from `simtoolreal`, keep proper MIT attribution in repo docs/license notices.
