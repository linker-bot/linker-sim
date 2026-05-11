# PR #2 progress log

**Status:** PR #2a and PR #2b both landed. Runtime backbone + tasks +
recorder/replayer + hydra + legacy deletion all in. Composed
workstations now carry `gain_profiles` in meta/manifest.

**Audience:** implementers picking up PR #3 (bimanual) or a second
backend. This doc captures the design decisions, file layout, dev
workflow, and explicitly-deferred items so future work doesn't have
to reverse-engineer them.

---

## What PR #2 is

Runtime scaffolding so every controller / task / recorder is written
against sim-agnostic protocols:

### PR #2a (landed)

- **`Robot` + `SimBackend` + `RigidBody` Protocols** in
  `sim/backends/base.py`. Torch tensors at the boundary (D10).
  Role-centric joint access.
- **Isaac backend** that composes `SimulationContext` +
  `InteractiveScene` directly — no `DirectRLEnv` / `ManagerBasedRLEnv`
  subclass (D9).
- **MuJoCo backend stubs** that satisfy the Protocol but raise
  `NotImplementedError("... blocked on PR #1b MJCFs")` on construction.
- **Controllers** (OSC, JointPD, IK) that own their gain profile via
  `robot.write_gains()` on `attach`.
- **`BaseEnv`** (gym-style) owning decimation, reset bookkeeping, and
  action slicing.

### PR #2b (landed)

- **`Task` Protocol** (`sim/tasks/base.py`) + two concrete tasks:
  **`ReachTask`** (sampled EE target, shaped distance reward, success
  held for N steps per `docs/target_spec.md` §5) and
  **`PickPlaceTask`** (staged grasp/lift/place with a pre-declared
  rigid body).
- **`RigidBody` Protocol** + `IsaacRigidBody` adapter; `IsaacBackendCfg`
  gains `rigid_bodies: dict[str, RigidBodySpec]` so tasks can pre-declare
  task objects at scene-build time.
- **`Recorder`** (`sim/io/recorder.py`) with three sinks: `null_sink`
  (discard), `JsonlSink` (one JSONL per episode), `LeRobotSink`
  (parquet; opt-in via `[lerobot]` extra). Recorder is
  `BaseEnv`-orthogonal — the driver wires it in.
- **`Replayer`** (`sim/io/replayer.py`) with `action_replay` /
  `state_inject` modes; reads `JsonlSink` output back.
- **Hydra config tree** at `sim/configs/` + `scripts/run.py` and
  `scripts/replay.py` as the single entry points.
  `python scripts/run.py task=reach recorder=jsonl` is the canonical
  smoke command.
- **`gain_profiles` lifted** from `loaders.py` into component
  `meta.yaml`; composer writes them into manifest; `OscController`
  prefers `handle.gain_profiles[role][name]` via its `gain_profile`
  cfg field. Old hardcoded constants gone entirely.
- **Legacy retirement**: `sim/envs/test/`, `sim/assets/*`,
  `tools/migrate_ar5_l6.py`, `assets/urdf/`, `sim/envs/test_osc/scene_cfg.py`,
  and the `TestOscRLEnv` shim are all gone. `spawn_osc_scene.py` and
  `gain_tuner_osc.py` migrated to the new backbone.

## Design decisions and why

Extends PR #1's D-series.

### D11. Controllers own their gain profile, loaders don't (PR #2a)

`sim/backends/isaac/loaders.py` used to contain a hardcoded
`control_mode="osc" -> kp=150, kd=8` branch. That coupled gain
selection to spawn-time.

PR #2a moves that concern to the controller: `OscController.attach()`
calls `robot.write_gains("arm", stiffness=..., damping=...)`. PR #2b
takes it further — the gains come from
`handle.gain_profiles[role][cfg.gain_profile]` by default; literal
values on `OscControllerCfg` are the fallback. Data-driven.

### D12. `Robot` is a thin adapter, not an owner

`IsaacRobot` wraps an `Articulation` handle. It caches per-role joint
index tensors at construction and routes state reads / command writes
straight to the articulation. No shadow state, no batching layer.

### D13. `BaseEnv` doesn't inherit from any IsaacLab env class

`BaseEnv` is a plain class with a gym-style
`reset/step` loop. It drives a `SimBackend`, dispatches action slices
to per-role controllers, and calls a `Task` for obs/reward/done. No
`DirectRLEnv`, no `ManagerBasedRLEnv`.

### D14. Fakes, not Isaac, in tests

All 15 tests under `tests/` drive backends / envs / controllers / tasks
through fake objects. None spawn a SimulationApp. The tradeoff: we
lose physics-level parity testing in CI, but we get a suite that runs
in 1s on any Python env. Isaac-level smoke is a manual step
(`python scripts/run.py`).

### D15. No `__init__.py` re-exports of Isaac-pulling modules

`sim/controllers/__init__.py` re-exports only `Controller` (the
Protocol). `OscController` etc. must be imported from their concrete
module. Same pattern for `sim/tasks/__init__.py`. Non-Isaac tools can
`import sim.controllers.base` without pulling in USD.

### D16. Recorder is orthogonal to `BaseEnv` (PR #2b)

`BaseEnv.step` returns its tensors; the driver (`scripts/run.py` or a
notebook) decides whether to call `recorder.record_step(...)`. This
keeps `BaseEnv` clean and lets a consumer bolt on multiple sinks
without touching the env.

Sink contract: `sink(episode_id: int, frame_idx: int, frame: dict) -> None`.
Recorder tracks per-env episode ids + frame indices and forwards
individual env frames.

### D17. Pre-declare rigid bodies, don't dynamically spawn (PR #2b)

IsaacLab's `InteractiveScene` wants all assets declared on the scene
cfg before the first `sim.reset()`. We respect that: tasks read
objects from `backend.rigid_bodies[name]` which were created from
`IsaacBackendCfg.rigid_bodies: dict[str, RigidBodySpec]` at scene
construction. No runtime `spawn_rigid(cfg)` — if you need a new
object type, add it to the cfg and restart.

*Why:* Dynamic spawn after `sim.reset()` in IsaacLab requires direct
USD stage manipulation and bypasses the scene cache. Not worth the
complexity for the MVP; add later if a task needs truly dynamic
object count.

### D18. Hydra configs use `_target_` instantiation (PR #2b)

`sim/configs/controller/osc.yaml` lists controllers with
`_target_: sim.controllers.osc.OscController` and a nested `cfg:`.
`scripts/run.py` uses `hydra.utils.instantiate(entry)` per entry.

*Why:* Uniform override syntax (`controller.entries.0.cfg.action_scale_pos=0.1`
works), and adding a new controller is a one-line YAML edit — no
Python dispatch table to update.

---

## File map

```
sim/
  backends/
    base.py                    # Robot + RigidBody + SimBackend Protocols
    isaac/
      backend.py               # IsaacSimBackend + _SingleRobotSceneCfg + IsaacRigidBody
      robot.py                 # IsaacRobot adapter
      loaders.py               # to_articulation_cfg (control_mode deprecated)
    mujoco/                    # Protocol-satisfying stubs
      __init__.py, backend.py, robot.py
  controllers/
    __init__.py                # exposes Controller Protocol only
    base.py                    # Controller Protocol
    osc.py                     # OscController (reads handle.gain_profiles)
    joint_pd.py                # JointPDController
    ik.py                      # DLS IK
  envs/
    base.py                    # BaseEnv (gym-style)
    test_osc/
      spawn_osc_scene.py       # manual smoke over IsaacSimBackend
      gain_tuner_osc.py        # OSC hot-reload tuner over new backbone
      osc_gains.json           # tuner state file
  tasks/
    __init__.py
    base.py                    # Task Protocol
    reach.py                   # ReachTask
    pick_place.py              # PickPlaceTask
  io/
    __init__.py
    recorder.py                # Recorder + null_sink / JsonlSink / LeRobotSink
    replayer.py                # Replayer + ReplayEpisode
  configs/
    config.yaml                # top-level defaults list
    backend/{isaac,mujoco}.yaml
    robot/{ar5_l6_bench,ar5_l6_bench_right}.yaml
    controller/{osc,joint_pd}.yaml
    task/{reach,pick_place}.yaml
    recorder/{null,jsonl,lerobot}.yaml
  registry.py                  # + gain_profiles field

scripts/
  run.py                       # hydra entrypoint
  replay.py                    # hydra replay entrypoint

tools/composer/
  schemas.py                   # + gain_profiles on ComponentMeta and Manifest
  compose.py                   # populates manifest.gain_profiles

tests/                         # 15 tests, no Isaac required
  test_protocols.py
  test_base_env.py
  test_reach_task.py           # NEW (2b)
  test_recorder_roundtrip.py   # NEW (2b)

assets/components/
  arms/ar5/meta.yaml           # + gain_profiles: {joint, osc}
  hands/linkerhand_l6/meta.yaml # + gain_profiles: {joint}

assets/workstations/*/manifest.yaml  # regenerated with gain_profiles
```

Deleted in PR #2b: `sim/envs/test/`, `sim/assets/`,
`sim/envs/test_osc/scene_cfg.py`, `sim/envs/test_osc/osc_rl_env.py`,
`tools/migrate_ar5_l6.py`, `assets/urdf/`.

## Runtime dataflow

```
scripts/run.py  (hydra)
  ├── IsaacBackendCfg(workstations, rigid_bodies, dt, device, …)
  ├── IsaacSimBackend  ──▶ SimulationContext + InteractiveScene
  │     ├── robots["robot"]   = IsaacRobot   (wraps Articulation)
  │     └── rigid_bodies["cube"] = IsaacRigidBody  (wraps RigidObject)
  │
  ├── [OscController(arm), JointPDController(hand)]  # controllers
  │     .attach(robot)  → caches joint ids, writes gain profile
  │
  ├── ReachTask | PickPlaceTask                      # task
  │     .observe / .reward / .done
  │
  ├── BaseEnv(backend, controllers, task)
  │     .reset(seed) → (obs, info)
  │     .step(action) → (obs, reward, terminated, truncated, info)
  │
  └── Recorder(sink=JsonlSink|LeRobotSink|null_sink)
        .record_step(obs, action, reward, terminated, truncated)
```

## Dev workflow

### One-shot smoke (OSC reach rollout)

```bash
python scripts/run.py
# defaults: backend=isaac, robot=ar5_l6_bench, controller=osc, task=reach, recorder=null
```

### Pick-and-place with JSONL recording

```bash
python scripts/run.py task=pick_place recorder=jsonl max_steps=500
# writes ./outputs/<date>/<time>/episodes/episode_000000.jsonl
```

### Replay an episode

```bash
python scripts/replay.py +episode=/path/to/episode_000000.jsonl
# or with state injection:
python scripts/replay.py +episode=... +mode=state_inject
```

### Run the tests

```bash
/home/zhy/opt/IsaacLab/env_isaaclab/bin/python -m pytest tests/
# 15 passed in < 1s
```

### OSC gain hot-reload tuner

```bash
python sim/envs/test_osc/gain_tuner_osc.py --num_envs 1 --robot_side left
# edit sim/envs/test_osc/osc_gains.json in another terminal; changes
# reload automatically.
```

### Adding a new task

1. Create `sim/tasks/<name>.py` with a class implementing the `Task`
   Protocol (see `sim/tasks/reach.py` for the minimal example).
2. Add `sim/configs/task/<name>.yaml` with a `_target_` pointing at
   your class and any task-level overrides.
3. Run: `python scripts/run.py task=<name>`.

### Adding a new controller

1. Create `sim/controllers/<name>.py` implementing the `Controller`
   Protocol.
2. Add an entry in `sim/configs/controller/<bundle>.yaml` (or make a
   new bundle).
3. Run: `python scripts/run.py controller=<bundle>`.

### Adding a new gain profile to a component

1. Edit `assets/components/<kind>/<name>/meta.yaml`:
   ```yaml
   gain_profiles:
     joint:  {stiffness: 1000, damping: 4}
     osc:    {stiffness: 150,  damping: 8}
     soft:   {stiffness: 50,   damping: 4}
   ```
2. Recompose: `python -m tools.composer.compose assets/workstations/<ws>`.
3. Reference it: `OscControllerCfg(gain_profile="soft")`.

## Known sharp edges

### Single-robot scenes only

`IsaacSimBackend` still enforces one entry in `cfg.workstations`.
Multi-robot (bimanual) is PR #3. The Protocol is ready —
`backend.robots` is a dict keyed by robot name — only the scene-cfg
builder needs updating.

### `OperationalSpaceController` is still an IsaacLab import

`sim/controllers/osc.py` delegates the OSC math to
`isaaclab.controllers.OperationalSpaceController`. A MuJoCo-side
`OscController` needs a reimplementation. Lands with a real MuJoCo
backend (post PR #1b).

### Pick-place uses a single pre-declared cube

`PickPlaceTask` expects `backend.rigid_bodies["cube"]`. Multi-object
tasks need to pre-declare each on `IsaacBackendCfg.rigid_bodies`. No
runtime `spawn_rigid(...)`; see D17.

### Cube pose is in world frame minus `env_origin`

`PickPlaceTask` approximates the cube pose in the robot root frame by
subtracting `env_origins`. That's exact for fixed-base workstations
with root at the env origin (which we have). Floating-base robots
would need an actual frame transform.

### `_tcp_to_wrist` assumption now lives in `handle.frames`

PR #2a migrated the `_tcp`/`_link7` naming hack into
`IsaacRobot._resolve_frame`. Downstream tasks can request `"arm:tool0"`
or `"arm:wrist"` (if the component declares a wrist mount frame).
AR5 currently only declares `tool0`; `wrist` is a soft TODO on the
component author.

### Phantom-link warnings from URDF import

Unchanged from PR1_PROGRESS.md. ~20 LoC composer fix; can land any
time.

### Hydra `_target_` + AppLauncher timing

`scripts/run.py` does `AppLauncher(...)` AFTER hydra resolves the
config but BEFORE any `sim/backends/isaac/...` import. Re-ordering
these imports breaks the SimulationApp lifecycle. Keep the lazy
imports inside `_run_isaac` if you refactor.

### LeRobotSink schema is minimal

We write `observation.state`, `action`, `reward`, `terminated`,
`truncated`, `episode_index`, `frame_index`, `timestamp`. Rich
LeRobot datasets include `observation.images.<cam>`, `task_index`,
etc. Map them in on the caller side; schema evolution is out of scope
per D7.

---

## Deferred work

### PR #1b — MJCF authoring (independent)

Unblocks `sim/backends/mujoco/` for real. Both pick_place and reach
will work against MuJoCo once the backend lands.

### PR #3 — bimanual workstation

`ar5_l6_bench_bimanual` recipe + manifest schema extension
(`ee_links: dict[role, str]`) + `IsaacSimBackend` multi-robot scene
builder. The runtime side is ready (controllers can target `arm_left`
/ `arm_right` roles independently).

### Composer cleanup (deferred from PR #1)

Phantom-link warnings; ~20 LoC fix in `tools/composer/urdf_ops.py`
(stub inertials + empty visuals for `world` / mount links). Cosmetic
but loud.

### Wrist mount frame on arm components

AR5's `meta.yaml` declares `tool0` but not `wrist` as a named mount
frame. Add `mount_frames.wrist: {parent: "AR5_5_07{V}_W4C4A2_link7"}`
to remove the ambient `_link7` assumption.

---

## Where to start next

1. **PR #1b** — component MJCF authoring. Unblocks the MuJoCo backend
   stubs.
2. **PR #3** — bimanual workstation. **Landed** — see
   [PR3_PROGRESS.md](PR3_PROGRESS.md).
3. **Agent training** — the backbone now supports gym-style RL. Plug
   an SB3 / skrl / CleanRL agent in around `BaseEnv` and go.
