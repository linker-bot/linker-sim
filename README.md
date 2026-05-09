# dex-tool-rl

RL simulation workspace for AR5 + Linkerhand L6 robots using IsaacLab/Isaac Sim.

## What this repo contains

- A minimal smoke-test simulation scene for AR5_L6 robot assets.
- Scene and robot asset configuration helpers.
- Early-stage task spec notes for future RL task/reward design.

## Current status

This repository is in active setup/prototyping stage. The default control pipeline is now OSC (end-effector pose control) under `sim/envs/test_osc`.

## Project layout

- `sim/assets`: robot and scene asset configs (URDF-backed `ArticulationCfg`/`AssetBaseCfg`)
- `sim/envs/test_osc`: OSC-first scene runners, gain tuner, and optional RL env
- `sim/envs/test`: legacy joint-space test env (deprecated)
- `docs/target_spec.md`: deterministic interface template for RL task definition
- `docs/installation.md`: setup guide (including local IsaacLab clone workflow)

## Installation

Use the setup guide in `docs/installation.md` for:

- `uv` + `env_isaaclab` environment setup (single shared env)
- IsaacLab clone at a canonical external location (e.g. `~/opt/IsaacLab/`)
- Installing this repo's packages via `uv pip install -e '.[tools]'`
- first-run troubleshooting

## Quick start

After completing `docs/installation.md`, run the OSC smoke test:

```bash
python sim/envs/test_osc/spawn_osc_scene.py --num_envs 1 --robot_side left
```

Optional dual-arm smoke test:

```bash
python sim/envs/test_osc/spawn_osc_scene.py --num_envs 1 --robot_side both
```

Legacy joint-space smoke test remains available but deprecated:

```bash
python sim/envs/test/spawn_scene.py --num_envs 1 --robot_side left
```

### Tuning joint stiffness / damping (Gain Tuner)

Isaac Sim’s **Gain Tuner** adjusts PhysX drive **stiffness** and **damping** on the articulation. This repo enables it and spawns the same test scene as the smoke test:

```bash
python sim/envs/test/gain_tuner_scene.py --num_envs 1 --robot_side left
```

Do not use `--headless`. The script defaults to a **passive command stream** mode so Isaac Lab does not overwrite per-step joint targets while tuning. In the app, open the Gain Tuner (menu path may vary by version; see [Isaac Sim joint tuning](https://docs.isaacsim.omniverse.nvidia.com/latest/robot_setup_tutorials/joint_tuning.html)), select the robot, and tune.

This repo now supports **per-joint PD arrays** (simtoolreal-style):

- Static defaults live in `sim/assets/robots.py` as per-joint lists/maps.
- Runtime-editable gains live in `sim/envs/test/joint_gains.json`.
- Optional runtime feed-forward torque offsets can be added in the same JSON under `offset`.

While `gain_tuner_scene.py` is running, edits to `joint_gains.json` are hot-reloaded automatically (default every `0.5s`) and applied in-place to the active articulation, so you do not need to restart Isaac Sim for stiffness/damping changes.

To print realtime joint effort estimates in the console while tuning:

```bash
python sim/envs/test/gain_tuner_scene.py --num_envs 1 --robot_side left --print_joint_effort
```

Optional controls:
- `--joint_effort_print_hz 5.0` (print rate)
- `--joint_effort_env_id 0` (which env to print)
- `--no-joint_effort_arm_only` (print hand joints too)

### Tuning OSC gains (Runtime Hot Reload)

Use the OSC tuner for end-effector-space controller tuning (not joint-drive PD tuning):

```bash
python sim/envs/test_osc/gain_tuner_osc.py --num_envs 1 --robot_side left
```

Runtime tuning file:
- `sim/envs/test_osc/osc_gains.json`

The file is auto-created on first run and hot-reloaded while the script is running.

To pass through extra Kit flags (for example another extension), use Isaac Lab’s `--kit_args` as documented for `AppLauncher`.

Expected behavior:

- Isaac Sim app launches.
- Scene spawns robot(s) and workstation table.
- Console prints setup/reset logs such as `[INFO] Test scene setup complete.`

## Development notes

- Keep generated artifacts out of git (`__pycache__`, virtual envs, logs).
- Put feature work on dedicated branches and commit before switching branches.
- IsaacLab lives outside this repo (e.g. `~/opt/IsaacLab/`) and is shared across projects; see [docs/installation.md](docs/installation.md).
