# Usage guide

Practical, command-first reference for everything you can actually *run*
in this repo. For installation see [installation.md](installation.md);
for the asset/composer model see [urdf_assets_infra.md](urdf_assets_infra.md);
for MJCF authoring see [component_mjcf_authoring.md](component_mjcf_authoring.md).

All commands assume the IsaacLab venv is active and you are at the repo
root:

```bash
source ~/opt/IsaacLab/env_isaaclab/bin/activate
cd /path/to/dex-tool-rl
```

If ROS 2 is sourced in your shell rc, prefix Python commands with
`env -u PYTHONPATH -u AMENT_PREFIX_PATH …` to keep `lark` / pytest happy.

---

## 1. Two entrypoints, one config tree

| Entrypoint                | Config root                | What it does                                       |
|---------------------------|----------------------------|----------------------------------------------------|
| [scripts/run.py](../scripts/run.py)         | [sim/configs/config.yaml](../sim/configs/config.yaml)  | Backend + controller + task + (optional) recorder rollout. |
| [scripts/replay.py](../scripts/replay.py)   | [sim/configs/replay.yaml](../sim/configs/replay.yaml)  | Replay external real-robot telemetry through a backend. No controllers, no task, no `BaseEnv`. |

Both are [Hydra](https://hydra.cc) entrypoints. Override anything on the
CLI by setting `group=name` (config group) or `dotted.path=value`
(direct override). Output dirs land under `outputs/YYYY-MM-DD/HH-MM-SS/`.

Config groups live under [sim/configs/](../sim/configs/):

- `backend/` — `isaac.yaml`, `mujoco.yaml`
- `robot/` — Hydra wrapper around a workstation name
- `controller/` — `joint_pd_bimanual`, `osc_bimanual`, `ik_pose_bimanual`
- `task/` — `bimanual_reach`, `bimanual_reach_ikpose`
- `recorder/` — `disabled`, `jsonl`, `lerobot`
- `source/` — replay sources (e.g. `data_collection`)

---

## 2. Run a rollout in Isaac Sim

Default: `backend=isaac`, `robot=ar5_l6_bench_bimanual`, `controller=osc_bimanual`,
`task=bimanual_reach`, `recorder=disabled`, `policy=zeros`.

```bash
# Smoke: bimanual reach with OSC, holding the default pose.
python scripts/run.py

# Choose another workstation and exercise both arms with random walk.
python scripts/run.py robot=lkls73_i1_bimanual policy=random_walk

# Headless, capped run — useful in CI / smoke tests.
python scripts/run.py headless=true max_steps=500

# Multi-env (vectorised).
python scripts/run.py num_envs=16 max_steps=200 headless=true
```

Hotkeys (windowed mode): press `R` in the viewport to reset all envs.
Close the window to exit.

Common knobs (defined in [sim/configs/config.yaml](../sim/configs/config.yaml)):

| Key                       | Default | Meaning                                                |
|---------------------------|---------|--------------------------------------------------------|
| `num_envs`                | 1       | Parallel envs.                                         |
| `env_spacing`             | 2.5     | Metres between envs (Isaac only).                      |
| `decimation`              | 4       | Physics steps per `env.step`.                          |
| `episode_length_s`        | 8.0     | Auto-truncate threshold.                               |
| `reset_joint_noise_scale` | 0.02    | Per-joint noise on reset.                              |
| `max_steps`               | 0       | `0` = run until the window closes.                     |
| `headless`                | false   | No viewport.                                           |
| `device`                  | cuda:0  | Torch device for tensors and Isaac sim.                |
| `policy`                  | zeros   | `zeros` (hold default) or `random_walk` (smoke).       |

---

## 3. Run a rollout in MuJoCo

The MuJoCo backend is CPU-only and does not support `rigid_bodies` (so
`task=pick_place`-style scenes are Isaac-only). Everything else works:

```bash
# Bimanual reach + joint PD + zeros, viewport.
python scripts/run.py backend=mujoco controller=joint_pd_bimanual \
    task=bimanual_reach policy=zeros max_steps=200

# Headless requires max_steps>0 (no viewport loop to terminate on).
python scripts/run.py backend=mujoco headless=true max_steps=400 \
    controller=joint_pd_bimanual task=bimanual_reach

# IK absolute-pose control (matched task: bimanual_reach_ikpose).
python scripts/run.py backend=mujoco controller=ik_pose_bimanual \
    task=bimanual_reach_ikpose recorder=jsonl
```

Hotkeys (windowed mode): press `R` in the MuJoCo viewport to reset.

---

## 4. Replay real-robot telemetry through MuJoCo / Isaac

Use [scripts/replay.py](../scripts/replay.py). It reads a
`ReplaySource` (currently `TelemetryNpzSource`), drives the workstation
with `set_joint_position_target` directly, and bypasses controllers,
tasks, and `BaseEnv`. Hand columns are decoded via per-joint mappers
(see [sim/io/replay/hands.py](../sim/io/replay/hands.py); the Linker L6
decoder is a placeholder linear map — see TODO inline).

```bash
# Real-robot recording at episode_000004/telemetry.npz, MuJoCo viewport,
# paced to 30 Hz wall-clock.
python scripts/replay.py robot=a7_lite_dc source=data_collection

# Headless smoke: cap to 200 frames, no realtime pacing.
python scripts/replay.py robot=a7_lite_dc source=data_collection \
    headless=true realtime=false max_frames=200

# Same data through Isaac (GPU).
python scripts/replay.py backend=isaac device=cuda:0 \
    robot=a7_lite_dc source=data_collection
```

Hotkeys (MuJoCo windowed mode): press `Q` in the viewport to stop.

### Adding a new recording

1. Drop the `.npz` somewhere readable. The default contains a `qpos`
   key shaped `(T, N)`.
2. Author `sim/configs/source/<your_name>.yaml` describing the column
   layout. Template (annotated) is
   [sim/configs/source/data_collection.yaml](../sim/configs/source/data_collection.yaml).
   Each role's `cols: [start, end)` slice **must** match the
   workstation's actuated-joint count for that role.
3. Run:
   ```bash
   python scripts/replay.py robot=<workstation> source=<your_name>
   ```

Replay knobs (in [sim/configs/replay.yaml](../sim/configs/replay.yaml)):
`realtime` (pace to `source.hz`), `max_frames` (clip), `headless`,
`device`.

### Inspecting a recording

```bash
# Dump arm joint trajectories to CSV + line-plot PNG (no replay).
python scripts/dump_arm_telemetry.py episode_000004
```

This is a standalone diagnostic; it does not touch the simulator.

---

## 5. Compose a workstation URDF / MJCF

A workstation is `recipe.yaml` → `workstation.urdf` + `workstation.mjcf`
+ `manifest.yaml` (manifest is the single source of truth the runtime
reads). Recipes live under [assets/workstations/](../assets/workstations/);
components under [assets/components/](../assets/components/).

### Compose

```bash
# One workstation.
python -m tools.composer.compose assets/workstations/a7_lite_dc

# All of them.
for ws in assets/workstations/*/; do
    python -m tools.composer.compose "$ws"
done
```

Output: `workstation.urdf`, `workstation.mjcf` (if every component
ships an MJCF), and `manifest.yaml`. Commit all three.

### Validate

```bash
# Per-component MJCF sanity (run before composing).
python -m tools.validate_component_mjcf assets/components/arms/a7_lite/variants/left
python -m tools.validate_component_mjcf assets/components/arms/a7_lite/variants/right
python -m tools.validate_component_mjcf assets/components/bases/a7_lite_torso/variants/default

# Workstation: 12 checks (manifest hashes, URDF kinematics, mesh
# resolution, drift, MJCF parity at 1e-5 m / 1e-5 rad).
python tools/validate_workstation.py assets/workstations/a7_lite_dc
```

### Drift gate (CI)

Catches recipe/component edits that forgot to re-commit generated files:

```bash
# All workstations.
bash tools/ci/check_drift.sh

# Single workstation.
bash tools/ci/check_drift.sh a7_lite_dc
```

Exit 0 = clean, 1 = drift.

### Inspect what the runtime sees

```bash
python tools/registry_show.py                  # list workstations
python tools/registry_show.py a7_lite_dc       # dump roles, joints, frames
```

### Adding a new workstation

1. Pick or author components under `assets/components/{arms,bases,hands}/`.
   Each ships `meta.yaml` + per-variant `<kind>.urdf` + `<kind>.mjcf` +
   `meshes/`. See [urdf_assets_infra.md](urdf_assets_infra.md) and
   [component_mjcf_authoring.md](component_mjcf_authoring.md).
2. Author `assets/workstations/<name>/recipe.yaml`.
3. Compose, validate, commit.
4. Add a thin Hydra wrapper at `sim/configs/robot/<name>.yaml`:
   ```yaml
   # @package _global_
   robot:
     workstation_name: <name>
     role_name: robot
     rigid_bodies: {}
   ```
5. Smoke: `python scripts/run.py robot=<name> backend=mujoco controller=joint_pd_bimanual task=bimanual_reach max_steps=200`.

---

## 6. PD / OSC gain tuning

Three places gains live. Edit the right one for what you want to change.

### a) Component defaults (per role, per arm — affects every workstation that uses it)

[meta.yaml](../assets/components/arms/lkls73_arm/meta.yaml) per
component:

```yaml
default_gains:
  stiffness: 1000
  damping: 4
gain_profiles:
  joint:  { stiffness: 1000, damping: 4 }   # used when controller=joint_pd_*
  osc:    { stiffness: 150,  damping: 8 }   # used when controller=osc_*
```

After editing: re-compose the workstation (`python -m tools.composer.compose …`)
and commit the regenerated `manifest.yaml` / `workstation.urdf`.

### b) Controller-level overrides (per controller config)

`stiffness` and `damping` on `JointPDControllerCfg` override the
manifest-level values for that role at runtime:

```yaml
# sim/configs/controller/joint_pd_bimanual.yaml
- _target_: sim.controllers.joint_pd.JointPDController
  cfg:
    role: arm_left
    action_scale: 0.25
    stiffness: 800
    damping: 6
```

Or override on the CLI:

```bash
python scripts/run.py controller=joint_pd_bimanual \
    'controller.entries.0.cfg.stiffness=800' \
    'controller.entries.0.cfg.damping=6'
```

OSC has its own block — `actuator_stiffness`, `actuator_damping`,
`stiffness` (task-space), `damping_ratio`, `nullspace_stiffness`,
`nullspace_damping_ratio` — see
[sim/configs/controller/osc_bimanual.yaml](../sim/configs/controller/osc_bimanual.yaml).

### c) MuJoCo `<position>` actuator gains (per-component MJCF)

MuJoCo bakes gains into actuators at model-load. Edit the per-component
MJCF (e.g. [arm.mjcf](../assets/components/arms/lkls73_arm/variants/left/arm.mjcf)):

```xml
<position name="L1_Joint_act" joint="L1_Joint" kp="1000" kv="4" ctrlrange="-3.752 2.181"/>
```

Then per-component validate, recompose, validate, commit. Component
gains here should match the manifest's `default_gains` to keep
URDF↔MJCF behaviour consistent.

### d) OSC hot-reload tuner (Isaac only, legacy path)

For interactive task-space gain tuning without restarting the sim:

```bash
python sim/envs/test_osc/gain_tuner_osc.py --num_envs 1 --workstation ar5_l6_bench_bimanual
```

While the script runs, edit
[sim/envs/test_osc/osc_gains.json](../sim/envs/test_osc/osc_gains.json) — changes
hot-reload (default 0.5 s). Tuned values can then be promoted to
`sim/configs/controller/osc_bimanual.yaml`. Do not run with
`--headless`; the tuner needs a viewport.

---

## 7. Recording episodes

Add a recorder to `scripts/run.py`:

```bash
# JSONL per-env episode files, one row per env.step.
python scripts/run.py recorder=jsonl max_steps=400

# LeRobot-format dataset at 30 fps.
python scripts/run.py recorder=lerobot max_steps=400
```

Recordings land under `outputs/YYYY-MM-DD/HH-MM-SS/episodes/`.
`recorder=disabled` (default) writes nothing.

---

## 8. Tests

```bash
# Pure-Python pytest gate (no GPU required).
env -u PYTHONPATH -u AMENT_PREFIX_PATH pytest tests/ -v
```

Full tiered pipeline (composer → registry → headless → reach → bimanual
→ recorder) is documented in [TEST_PIPELINE.md](TEST_PIPELINE.md).

---

## 9. Cheatsheet

```bash
# Isaac, default everything
python scripts/run.py

# Isaac, bimanual reach + OSC + recorded episodes
python scripts/run.py robot=lkls73_i1_bimanual recorder=jsonl max_steps=600

# MuJoCo, joint PD smoke
python scripts/run.py backend=mujoco controller=joint_pd_bimanual \
    task=bimanual_reach policy=zeros max_steps=200

# MuJoCo, IK absolute pose
python scripts/run.py backend=mujoco controller=ik_pose_bimanual \
    task=bimanual_reach_ikpose

# Replay real-robot data (MuJoCo)
python scripts/replay.py robot=a7_lite_dc source=data_collection

# Replay headless / clipped
python scripts/replay.py robot=a7_lite_dc source=data_collection \
    headless=true realtime=false max_frames=200

# Compose + validate everything
for ws in assets/workstations/*/; do python -m tools.composer.compose "$ws"; done
for ws in assets/workstations/*/; do python tools/validate_workstation.py "$ws"; done
bash tools/ci/check_drift.sh

# Inspect a registry handle
python tools/registry_show.py a7_lite_dc

# Dump arm trajectories from a recording
python scripts/dump_arm_telemetry.py episode_000004
```
