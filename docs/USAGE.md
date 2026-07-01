# Usage guide

Practical, command-first reference for everything you can actually *run*
in this repo. For installation see [installation.md](installation.md);
for the asset/composer model see [urdf_assets_infra.md](urdf_assets_infra.md);
for MJCF authoring see [component_mjcf_authoring.md](component_mjcf_authoring.md).

All commands assume the IsaacLab venv is active and you are at the repo
root:

```bash
source ~/opt/IsaacLab/env_isaaclab/bin/activate
cd /path/to/linker-sim
```

If ROS 2 is sourced in your shell rc, prefix Python commands with
`env -u PYTHONPATH -u AMENT_PREFIX_PATH …` to keep `lark` / pytest happy.

---

## 1. Two entrypoints, one config tree

| Entrypoint                | Config root                | What it does                                       |
|---------------------------|----------------------------|----------------------------------------------------|
| [scripts/run.py](../scripts/run.py)         | [linker_sim/configs/config.yaml](../packages/linker-sim/src/linker_sim/configs/config.yaml)  | Backend + controller + task + (optional) recorder rollout. |
| [scripts/replay.py](../scripts/replay.py)   | [linker_sim/configs/replay.yaml](../packages/linker-sim/src/linker_sim/configs/replay.yaml)  | Replay external real-robot telemetry through a backend. No controllers, no task, no `BaseEnv`. |

Both are [Hydra](https://hydra.cc) entrypoints. Override anything on the
CLI by setting `group=name` (config group) or `dotted.path=value`
(direct override). Output dirs land under `outputs/YYYY-MM-DD/HH-MM-SS/`.

Config groups live under [linker_sim/configs/](../packages/linker-sim/src/linker_sim/configs/):

- `backend/` — `isaac.yaml`, `mujoco.yaml`, `viser.yaml` (replay-only, browser visualisation)
- `robot/` — Hydra wrapper around a workstation name
- `controller/` — `joint_pd_bimanual`, `osc_bimanual` (stub), `ik_pose_bimanual`
- `task/` — `bimanual_reach_ikpose`
- `recorder/` — `disabled`, `jsonl`, `lerobot`
- `source/` — replay sources (e.g. `data_collection`)

---

## 2. Run a rollout in Isaac Sim

Default: `backend=isaac`, `robot=ar5_o6_bench_bimanual`, `controller=joint_pd_bimanual`,
`task=bimanual_reach_ikpose`, `recorder=disabled`, `policy=zeros`.

```bash
# Smoke: bimanual reach with joint PD, holding the default pose.
python scripts/run.py

# Choose another workstation and exercise both arms with random walk.
python scripts/run.py robot=lkls73_i1_o6_bimanual policy=random_walk

# Headless, capped run — useful in CI / smoke tests.
python scripts/run.py headless=true max_steps=500

# Multi-env (vectorised).
python scripts/run.py num_envs=16 max_steps=200 headless=true
```

Hotkeys (windowed mode): press `R` in the viewport to reset all envs.
Close the window to exit.

Shipped workstations (Hydra group `robot`):

- O6 hand (default class): `ar5_o6_bench_bimanual` (default),
  `ar5_08_o6_bench_bimanual`, `lkls73_i1_o6_bimanual`, `a7_lite_o6_dc`.
- L25 hand: `ar5_l25_bench_bimanual`, `ar5_08_l25_bench_bimanual`,
  `lkls73_i1_l25_bimanual`, `a7_lite_l25_dc`.
- L6 hand (legacy / parallel): `ar5_l6_bench_bimanual`,
  `lkls73_i1_bimanual`, `a7_lite_l6_dc`.

Common knobs (defined in [linker_sim/configs/config.yaml](../packages/linker-sim/src/linker_sim/configs/config.yaml)):

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
| `policy`                  | zeros   | `zeros` (hold default), `random_walk` (smoke), or `hold` (no controller writes — for live gain tuning). |

---

## 3. Run a rollout in MuJoCo

The MuJoCo backend is CPU-only and does not support `rigid_bodies` (so
`task=pick_place`-style scenes are Isaac-only). Everything else works:

```bash
# Bimanual reach + joint PD + zeros, viewport.
python scripts/run.py backend=mujoco controller=joint_pd_bimanual \
    task=bimanual_reach_ikpose policy=zeros max_steps=200

# Headless requires max_steps>0 (no viewport loop to terminate on).
python scripts/run.py backend=mujoco headless=true max_steps=400 \
    controller=joint_pd_bimanual task=bimanual_reach_ikpose

# IK absolute-pose control (controller=ik_pose_bimanual is the matched pair).
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
(see [linker_sim/io/replay/hands.py](../packages/linker-sim/src/linker_sim/io/replay/hands.py)).

> **Hand-decoder accuracy caveat.** The `linker_l6` and `linker_o6`
> decoders linearly interpolate the vendor's 0–255 byte command across
> each joint's `[lower, upper]` limit. The real Linker Hand calibration
> is non-linear and may invert travel direction on some joints, so
> arm tracking is faithful but **finger pose is approximate**. Replace
> these decoders with vendor curves (or a per-joint LUT) before
> claiming physical replay fidelity. Tracked inline as a TODO in
> [linker_sim/io/replay/hands.py](../packages/linker-sim/src/linker_sim/io/replay/hands.py).

```bash
# Real-robot recording at episode_000025/telemetry.npz, MuJoCo viewport,
# paced to 30 Hz wall-clock. The recording is not tracked in this repo —
# either drop it at the default path or override with `source.path=...`.
python scripts/replay.py robot=a7_lite_l6_dc source=data_collection

# Headless smoke: cap to 200 frames, no realtime pacing.
python scripts/replay.py robot=a7_lite_l6_dc source=data_collection \
    headless=true realtime=false max_frames=200

# Same data through Isaac (GPU).
python scripts/replay.py backend=isaac device=cuda:0 \
    robot=a7_lite_l6_dc source=data_collection

# Same data through the Viser browser visualiser (replay-only, no GPU
# needed). Open the URL printed at startup, default http://127.0.0.1:8080.
# Requires the [viser] install profile — see the Data-collection team
# section in the README; not compatible with the env_isaaclab venv.
python scripts/replay.py backend=viser robot=a7_lite_l6_dc source=data_collection
```

Hotkeys (MuJoCo windowed mode): press `Q` in the viewport to stop.

> **Viser is replay-only.** `ViserSimBackend` animates a URDF in the
> browser as joint targets stream in; `step()` is a no-op and dynamics
> methods (Jacobian, mass matrix, ee_pose_b, set_joint_effort) raise
> `NotImplementedError`. Use it for `scripts/replay.py` only — never
> with `scripts/run.py`. Teleop is deferred.

### Adding a new recording

1. Drop the `.npz` somewhere readable. The default contains a `qpos`
   key shaped `(T, N)`.
2. Author `sim/configs/source/<your_name>.yaml` describing the column
   layout. Template (annotated) is
   [linker_sim/configs/source/data_collection.yaml](../packages/linker-sim/src/linker_sim/configs/source/data_collection.yaml).
   Each role's `cols: [start, end)` slice **must** match the
   workstation's actuated-joint count for that role.
3. Run:
   ```bash
   python scripts/replay.py robot=<workstation> source=<your_name>
   ```

Replay knobs (in [linker_sim/configs/replay.yaml](../packages/linker-sim/src/linker_sim/configs/replay.yaml)):
`realtime` (pace to `source.hz`), `max_frames` (clip), `headless`,
`device`.

---

## 5. Compose a workstation URDF / MJCF

A workstation is `recipe.yaml` → `workstation.urdf` + `workstation.mjcf`
+ `manifest.yaml` (manifest is the single source of truth the runtime
reads). Recipes live under [assets/workstations/](../packages/linker-robot-assets/src/linker_robot_assets/assets/workstations/);
components under [assets/components/](../packages/linker-robot-assets/src/linker_robot_assets/assets/components/).

### Compose

```bash
# One workstation.
python -m linker_robot_assets.composer.compose packages/linker-robot-assets/src/linker_robot_assets/assets/workstations/a7_lite_l6_dc

# All of them.
for ws in packages/linker-robot-assets/src/linker_robot_assets/assets/workstations/*/; do
    python -m linker_robot_assets.composer.compose "$ws"
done
```

Output: `workstation.urdf`, `workstation.mjcf` (if every component
ships an MJCF), and `manifest.yaml`. Commit all three.

### Validate

```bash
# Per-component MJCF sanity (run before composing).
python -m linker_robot_assets.validate_component_mjcf packages/linker-robot-assets/src/linker_robot_assets/assets/components/arms/a7_lite/variants/left
python -m linker_robot_assets.validate_component_mjcf packages/linker-robot-assets/src/linker_robot_assets/assets/components/arms/a7_lite/variants/right
python -m linker_robot_assets.validate_component_mjcf packages/linker-robot-assets/src/linker_robot_assets/assets/components/bases/a7_lite_torso/variants/default

# Workstation: 14 checks (manifest hashes, URDF kinematics, mesh
# resolution, drift, MJCF parity at 1e-5 m / 1e-5 rad).
python -m linker_robot_assets.validate_workstation packages/linker-robot-assets/src/linker_robot_assets/assets/workstations/a7_lite_l6_dc
```

### Drift gate (CI)

Catches recipe/component edits that forgot to re-commit generated files:

```bash
# All workstations.
bash packages/linker-robot-assets/src/linker_robot_assets/ci/check_drift.sh

# Single workstation.
bash packages/linker-robot-assets/src/linker_robot_assets/ci/check_drift.sh a7_lite_l6_dc
```

Exit 0 = clean, 1 = drift.

### Inspect what the runtime sees

```bash
python -m linker_sim.tools.registry_show                  # list workstations
python -m linker_sim.tools.registry_show a7_lite_l6_dc       # dump roles, joints, frames
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
5. Smoke: `python scripts/run.py robot=<name> backend=mujoco controller=joint_pd_bimanual task=bimanual_reach_ikpose max_steps=200`.

---

## 6. PD / OSC gain tuning

Three places gains live. Edit the right one for what you want to change.

### a) Component defaults (per role, per arm — affects every workstation that uses it)

[meta.yaml](../packages/linker-robot-assets/src/linker_robot_assets/assets/components/arms/lkls73_arm/meta.yaml) per
component:

```yaml
default_gains:
  stiffness: 1000
  damping: 4
gain_profiles:
  joint:  { stiffness: 1000, damping: 4 }   # used when controller=joint_pd_*
  osc:    { stiffness: 150,  damping: 8 }   # used when controller=osc_*
```

After editing: re-compose the workstation (`python -m linker_robot_assets.composer.compose …`)
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
[linker_sim/configs/controller/osc_bimanual.yaml](../packages/linker-sim/src/linker_sim/configs/controller/osc_bimanual.yaml).

### c) MuJoCo `<position>` actuator gains (per-component MJCF)

MuJoCo bakes gains into actuators at model-load. Stiffness lives on
the actuator (`kp`); damping lives on the **joint** (`damping`
attribute) because MuJoCo integrates joint damping implicitly
(unconditionally stable), whereas actuator `kv` is explicit and can
diverge at high values. Edit the per-component MJCF (e.g.
[arm.mjcf](../packages/linker-robot-assets/src/linker_robot_assets/assets/components/arms/a7_lite/variants/left/arm.mjcf)):

```xml
<!-- Joint: damping here (implicit, stable) -->
<joint name="L1_Joint" ... damping="20" armature="0.01"/>

<!-- Actuator: kp here, kv=0 (damping is on the joint) -->
<position name="L1_Joint_act" joint="L1_Joint" kp="2000" kv="0" ctrlrange="-2.18 3.75"/>
```

Then per-component validate, recompose, validate, commit. Component
gains here should match the manifest's `default_gains` to keep
URDF↔MJCF behaviour consistent.

### d) Live PD gain tuner (any backend)

Use `policy=hold` + `+gain_tuner=true` to hot-reload joint PD gains
from a JSON file while the sim runs. The `hold` policy returns no
actions so the controller never writes targets — the robot holds its
current pose via the position actuators while you tweak gains.

```bash
# MuJoCo — live tune, file at /tmp/dex_pd_gains.json (default)
python scripts/run.py backend=mujoco controller=joint_pd_bimanual \
    task=bimanual_reach_ikpose policy=hold +gain_tuner=true

# Custom path
python scripts/run.py backend=mujoco controller=joint_pd_bimanual \
    task=bimanual_reach_ikpose policy=hold +gain_tuner=true \
    +gain_tuner_path=/tmp/my_gains.json
```

On first run the JSON is seeded from the workstation manifest's
`default_gains`. Edit it in another terminal — changes are picked up
every 0.5 s:

```json
{
  "arm_left":  { "stiffness": 2000, "damping": 20 },
  "arm_right": { "stiffness": 2000, "damping": 20 },
  "hand_left": { "stiffness": 200,  "damping": 4 },
  "hand_right":{ "stiffness": 200,  "damping": 4 }
}
```

Once tuned, promote the values to the component MJCF and meta.yaml
(sections a/c above). The JSON file is session-only and not tracked.

Implementation: [linker_sim/io/gain_watcher.py](../packages/linker-sim/src/linker_sim/io/gain_watcher.py).

### e) OSC controller (not implemented)

The OSC controller (`linker_sim/controllers/osc.py`) and its tuner
(`linker_sim/envs/test_osc/gain_tuner_osc.py`) are stubbed out — the previous
implementation was never validated. The Hydra config
`controller=osc_bimanual` still exists but will raise
`NotImplementedError` at runtime.

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

---

## 9. Cheatsheet

```bash
# Isaac, default everything
python scripts/run.py

# Isaac, bimanual reach + recorded episodes
python scripts/run.py robot=lkls73_i1_bimanual recorder=jsonl max_steps=600

# MuJoCo, joint PD smoke
python scripts/run.py backend=mujoco controller=joint_pd_bimanual \
    task=bimanual_reach_ikpose policy=zeros max_steps=200

# MuJoCo, IK absolute pose
python scripts/run.py backend=mujoco controller=ik_pose_bimanual \
    task=bimanual_reach_ikpose

# Replay real-robot data (MuJoCo)
python scripts/replay.py robot=a7_lite_l6_dc source=data_collection

# Replay headless / clipped
python scripts/replay.py robot=a7_lite_l6_dc source=data_collection \
    headless=true realtime=false max_frames=200

# Compose + validate everything
for ws in packages/linker-robot-assets/src/linker_robot_assets/assets/workstations/*/; do python -m linker_robot_assets.composer.compose "$ws"; done
for ws in packages/linker-robot-assets/src/linker_robot_assets/assets/workstations/*/; do python -m linker_robot_assets.validate_workstation "$ws"; done
bash packages/linker-robot-assets/src/linker_robot_assets/ci/check_drift.sh

# Inspect a registry handle
python -m linker_sim.tools.registry_show a7_lite_l6_dc

# Live PD gain tuning (MuJoCo, edit /tmp/dex_pd_gains.json while running)
python scripts/run.py backend=mujoco controller=joint_pd_bimanual \
    task=bimanual_reach_ikpose policy=hold +gain_tuner=true
```
