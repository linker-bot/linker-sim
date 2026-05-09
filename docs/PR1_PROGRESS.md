# PR #1 progress log

**Status:** PR #1a landed (composer + registry + Isaac-side migration of the
OSC smoke test). MJCF authoring (PR #1b) and broader runtime scaffolding
(PR #2) are queued.

**Audience:** implementers picking up later PRs. This doc captures the
design decisions, file layout, dev workflow, and explicitly-deferred
items so future work doesn't have to reverse-engineer them.

---

## What PR #1 is

Infrastructure to make the repo a multi-robot simulation backbone:

- **Decompose** monolithic per-robot URDFs into reusable components
  (arms, hands, bases, sensors).
- **Compose** them at author-time into per-workstation monolithic
  artifacts (one `workstation.urdf` + one `workstation.mjcf` per
  workstation variant) that Isaac Sim and MuJoCo load directly.
- **Load** composed artifacts at runtime through a thin sim-agnostic
  registry.

PR #1 does not include: the MuJoCo backend, the `SimBackend` / `Robot`
protocol layer, controllers written against those protocols, the data
recorder/replayer glue, or any cross-sim parity test. Those are PR #2+.

## Design decisions and why

Each entry links back to conversations that settled it; open them again
only if a new constraint breaks the original premise.

### D1. Author-time composition, not runtime
Source components live at `assets/components/…`. A Python composer
produces a committed monolithic `workstation.urdf` (and later `.mjcf`)
per workstation. The simulator loads the committed file; no runtime
splicing.

*Why:* MuJoCo needs one model root with a flat global namespace and a
single `meshdir`. Runtime multi-articulation assembly with weld
constraints degrades physics fidelity (PhysX residual at the splice,
MuJoCo equality-constraint drift) and breaks controllers that need a
coherent mass matrix / jacobian from base to fingertip.

### D2. Monolithic committed MJCF per workstation (no `<include>` at load)
The workstation's `workstation.mjcf` is a single self-contained file,
produced by a composer pass that textually inlines component MJCF bodies.
Authors never look at `<include>` resolution at sim load time.

*Why:* User preference for stability. What you load is exactly what's
in the file; no cross-file include surprises; external tooling that
doesn't understand `<include>` can still load the artifact.

### D3. No xacro
URDF composition is done with a ~300-line Python composer
(`tools/composer/`) using stdlib `xml.etree`. xacro is not a pipeline
dependency; component authors may use xacro locally to author a
component's own URDF, but the composer only consumes flat URDFs.

*Why:* Our job is *composition* (assemble N pre-authored parts at known
mount points), not *parameterization* (macros, repeated-finger
templates). xacro is the wrong tool for composition: it requires every
component to be re-authored as `<xacro:macro>`, it has poor error
messages, it adds a ROS tooling dependency, and it doesn't help with
MJCF anyway. Python is a single language for both URDF and MJCF
composition paths.

### D4. Role-prefixed names, not runtime remapping
Composer renames every link/joint/material/transmission/equality
reference with a role prefix (`arm_`, `hand_`, `base_`, `sensor_<name>_`).
Bimanual variants use `arm_left_` / `arm_right_`.

*Why:* MuJoCo has a flat global namespace. Two components with the same
link name (`base_link`) would collide without prefixing. Isaac via USD
could avoid this via prim-path addressing but we still prefix for
cross-stack consistency — the same names flow into the real-robot
driver, the LeRobot collector, and MoveIt.

### D5. Variants with template placeholders
Components that ship multiple variants (left/right) use a `{V}` or `{S}`
placeholder in meta.yaml link/joint names that expands per variant.
Meshes are pre-mirrored per variant. No runtime mirroring.

*Why:* AR5 and L6 already ship pre-mirrored meshes. Variant is an
author-level fact, not a runtime parameter.

### D6. Manifest as the runtime source of truth
The composer emits `workstation.yaml/urdf/mjcf + manifest.yaml`. The
manifest lists joints per role, frames, EE link, merged gains, and
component provenance (sha256). The registry (`sim/registry.py`) reads
the manifest and trusts it — schema validation is the composer's job.

*Why:* Separates generated content from authored content. Keeps the
runtime registry free of validation logic that would need to stay in
sync with the composer.

### D7. Data pipeline is out of scope for this repo
A LeRobot-compatible collector exists elsewhere. This repo provides:

- `Recorder` hook into `Env.step` that emits `(obs, action, info)` to
  the external collector (PR #2).
- `Replayer` that consumes LeRobot episodes in `action_replay` or
  `state_inject` modes (PR #2).

No storage format decisions, no dataset directory, no schema ownership.

### D8. MuJoCo vectorization is not in this repo
Per the finalized plan, MuJoCo is `B=1` per process. Parallel rollouts
happen at the trainer level (torchrun / Ray). No subprocess-based
`VectorMujocoBackend`.

### D9. Own `Env` base, don't subclass `ManagerBasedRLEnv`
Per the finalized plan, `sim/envs/base.py::BaseEnv` is gym.Env-style
and owned by this repo. Isaac path composes `SimulationContext` +
`Articulation` directly; it doesn't inherit from IsaacLab's
`ManagerBasedRLEnv`. This shrinks the IsaacLab API surface we depend on.

*Why:* We'd override `ManagerBasedRLEnv.step` entirely anyway; the
subclass buys nothing except coupling to IsaacLab manager churn.

### D10. Torch at the env boundary only
`Env.step(action) -> obs` uses torch `(B, ...)` tensors. Inside the
MuJoCo backend we use numpy; the conversion happens at the backend
boundary. Isaac stays torch-native (IsaacLab is torch-native; free).

---

## File map

```
pyproject.toml                   # project metadata + dep extras (tools, mujoco, isaac, dev, all)
assets/
  components/                    # reusable hardware pieces (new)
    arms/ar5/
      meta.yaml                  # contract the composer + registry read
      variants/
        {left,right}/
          arm.urdf               # flat, self-contained (arm only)
          arm.mjcf               # (PR #1b) hand-authored
          meshes/*.stl
    hands/linkerhand_l6/
      meta.yaml
      variants/{left,right}/{hand.urdf, hand.mjcf, meshes/*.STL}
    bases/bench_table/
      meta.yaml
      variants/default/{base.urdf, base.mjcf, meshes/*.STL}
  workstations/                  # composed, loadable units (new)
    ar5_l6_bench/                # left-side workstation
      recipe.yaml                # authored
      workstation.urdf           # generated, committed
      workstation.mjcf           # (PR #1b) generated
      manifest.yaml              # generated, what registry.load reads
    ar5_l6_bench_right/          # right-side workstation
      recipe.yaml
      workstation.urdf
      manifest.yaml
  urdf/                          # LEGACY; unchanged; loaded by legacy
                                 # scene_assets paths until PR #2 completes.
  scenes/                        # (reserved for PR #2) non-robot scene specs

tools/
  requirements.txt  # DELETED: merged into pyproject.toml's [tools] extra
  composer/
    __init__.py                  # COMPOSER_VERSION
    schemas.py                   # ComponentMeta, Recipe, Manifest dataclasses + YAML loaders
    determinism.py               # deterministic XML serialization
    urdf_ops.py                  # URDF composer: prefixing, merging, mount joints, mesh paths
    mjcf_ops.py                  # MJCF skeleton: availability check + NotImplementedError stub for PR #1b
    compose.py                   # CLI orchestrator; `python -m tools.composer.compose <ws_dir>`
  validate_workstation.py        # 8 checks per workstation; exit 0 = green
  registry_show.py               # CLI dump of registry.load(name)
  migrate_ar5_l6.py              # one-shot legacy split (ran once during PR #1a)
  ci/check_drift.sh              # CI gate — re-runs composer, fails on diff

sim/
  registry.py                    # read-only manifest loader; sim-agnostic
  backends/
    __init__.py
    isaac/
      __init__.py
      loaders.py                 # to_articulation_cfg(handle) -> ArticulationCfg
  assets/                        # LEGACY; kept for non-OSC envs until PR #2
    robots.py                    # hardcoded AR5 URDF paths + joint regexes
    scene_assets.py              # make_ar5_l6_*_robot_cfg, make_workspace_table_cfg
    __init__.py
  envs/
    test_osc/
      scene_cfg.py               # OscWorkstationSceneCfg (NEW) + legacy cfgs
      osc_rl_env.py              # now reads joint/body names from the handle
      spawn_osc_scene.py         # `--robot_side` maps to workstation name
      gain_tuner_osc.py          # TOUCHED: not yet migrated — still uses legacy cfgs
      osc_gains.json
    test/                        # legacy joint-space env; slated for deletion in PR #2

docs/
  component_mjcf_authoring.md    # contract for PR #1b MJCF authoring
  PR1_PROGRESS.md                # this file
```

## The composer pipeline

```
  components/arms/ar5/variants/left/arm.urdf        ──┐
  components/hands/linkerhand_l6/variants/left/hand.urdf ──┤
  components/bases/bench_table/variants/default/base.urdf ──┤
                                                            ├── compose.py ──┐
  workstations/ar5_l6_bench/recipe.yaml             ────────┘                │
                                                                             ▼
                                       workstations/ar5_l6_bench/workstation.urdf  (generated, committed)
                                       workstations/ar5_l6_bench/workstation.mjcf  (PR #1b)
                                       workstations/ar5_l6_bench/manifest.yaml     (generated, committed)
```

Composer operations per component:
1. Load flat URDF.
2. Rename every `<link name=>`, `<joint name=>`, `<material name=>`,
   `<transmission name=>` with the role prefix.
3. Rewrite every reference: `<parent link=>`, `<child link=>`,
   `<mimic joint=>`, `<mujoco><equality><joint joint1= joint2=>` so
   references still resolve after rename.
4. Rewrite `<mesh filename=>` paths from the URDF's own directory
   to a path relative to the workstation directory.
5. Merge into one `<robot>` with the recipe's mount fixed joints.
6. If `freeze_base` is set, add a `world` link + fixed joint to the
   base component's root link.
7. Serialize deterministically: sorted vector-attribute formatting,
   UTF-8 declaration, trailing newline.

## Runtime dataflow

```
registry.load(name)          # reads assets/workstations/<name>/manifest.yaml
  -> WorkstationHandle       # joints, mimic_joints, frames, ee_link, base_link, default_gains, components

Isaac path:
  sim.backends.isaac.loaders.to_articulation_cfg(handle, prim_path, control_mode)
    -> isaaclab.assets.ArticulationCfg

MuJoCo path (PR #1b):
  sim.backends.mujoco.loaders.to_mj_model(handle)
    -> mujoco.MjModel
```

The registry is pure — no sim imports. Each backend owns the conversion
from handle to its native cfg.

## Dev workflow

### Running the OSC smoke test
```bash
# Single-arm (composed workstation, unified articulation):
python sim/envs/test_osc/spawn_osc_scene.py --robot_side left
python sim/envs/test_osc/spawn_osc_scene.py --robot_side right
python sim/envs/test_osc/spawn_osc_scene.py --workstation ar5_l6_bench   # explicit

# Dual arm: temporarily unsupported — needs a bimanual workstation recipe.
```

### Recomposing after editing a component or recipe
```bash
python -m tools.composer.compose assets/workstations/ar5_l6_bench
python -m tools.composer.compose assets/workstations/ar5_l6_bench_right
# Or recompose everything:
for ws in assets/workstations/*/; do python -m tools.composer.compose "$ws"; done
```

### Validating a workstation
```bash
python tools/validate_workstation.py assets/workstations/ar5_l6_bench
```
Runs 8 checks: manifest self-consistency (recipe + component + URDF
hashes), URDF joint-count coverage, EE + mount-frame link resolution,
mesh path resolution, single kinematic tree, composer drift. Exit 0 =
green.

### CI drift check (every PR)
```bash
bash tools/ci/check_drift.sh
```
Iterates every `assets/workstations/*/`, re-runs the composer with
`--check-drift`, fails if committed artifacts differ from fresh.

### Inspecting what the registry returns
```bash
python tools/registry_show.py                  # list workstations
python tools/registry_show.py ar5_l6_bench     # dump the handle
```

### Adding a new component (e.g. a new arm)
1. Create `assets/components/arms/<name>/meta.yaml` with kind/variants/root_link/mount_frames/actuated_joints/default_gains.
2. Add `assets/components/arms/<name>/variants/<v>/arm.urdf` (flat) and
   `meshes/`.
3. Add `arm.mjcf` per [docs/component_mjcf_authoring.md](component_mjcf_authoring.md) (or defer; workstations work URDF-only).
4. No existing workstation needs changes — the component is available
   but not yet referenced.

### Adding a new workstation (new robot combination)
1. Create `assets/workstations/<name>/recipe.yaml` with components +
   mounts + optional `freeze_base`.
2. Run `python -m tools.composer.compose assets/workstations/<name>`.
3. Run `python tools/validate_workstation.py assets/workstations/<name>`.
4. Commit generated `workstation.urdf`, `manifest.yaml` (and `.mjcf`
   once components have MJCFs).

---

## Deferred work

### PR #1b — MJCF authoring
- Hand-author `assets/components/arms/ar5/variants/{left,right}/arm.mjcf`
  per [docs/component_mjcf_authoring.md](component_mjcf_authoring.md).
- Same for `hands/linkerhand_l6` and `bases/bench_table`.
- Flesh out `tools/composer/mjcf_ops.py::compose_mjcf` (currently
  `NotImplementedError`). Mirror URDF composer ops for MJCF: rename,
  merge `<asset>` / `<default>` / `<worldbody>` / `<actuator>` /
  `<equality>` / `<contact>` / `<sensor>` sections, dedupe mesh
  assets, splice mount bodies.
- Extend `tools/validate_workstation.py` with MJCF checks per
  authoring guide §9.
- Extend CI drift gate to cover `workstation.mjcf`.

### PR #2 — runtime backbone
Per the finalized plan:

- `sim/backends/base.py`: `SimBackend` + `Robot` `Protocol`s.
- `sim/backends/isaac/backend.py` + `robot.py`: concrete Isaac impl.
  Composes `SimulationContext` + `Articulation` directly (no
  `ManagerBasedRLEnv` subclass).
- `sim/controllers/{base,osc,joint_pd,ik}.py`: controllers written
  purely against the `Robot` protocol, reusing `WorkstationHandle` for
  joint lists + gains.
- `sim/envs/base.py`: `BaseEnv` (gym.Env-style) owning decimation +
  reset bookkeeping.
- `sim/tasks/{base,reach,pick_place}.py`: tasks own obs + reward + done.
- `sim/io/{recorder,replayer}.py`: LeRobot glue.
- Hydra configs for backend × robot × controller × task.
- Gain profiles: lift the hardcoded OSC override in
  `sim/backends/isaac/loaders.py` into a component `gain_profiles`
  section in `meta.yaml`; let recipes select a profile.

### PR #2 follow-up — `sim/assets/` retirement
`sim/assets/robots.py` and `sim/assets/scene_assets.py` still hold
legacy URDF paths + make_*_robot_cfg functions. Nothing new imports
them, but `sim/envs/test/` (joint-space legacy) and
`sim/envs/test_osc/gain_tuner_osc.py` still do.

Plan:
- Migrate `gain_tuner_osc.py` to `OscWorkstationSceneCfg`.
- Delete `sim/envs/test/` (joint-space env superseded by
  `sim/controllers/joint_pd.py` + generic `BaseEnv`).
- Delete `sim/assets/robots.py` + `sim/assets/scene_assets.py`.
- Delete `assets/urdf/` (legacy monolithic URDFs + meshes). Update
  `.gitignore` / `.gitattributes` accordingly.

### PR #3 (or later) — bimanual workstation
`--robot_side both` currently errors. To restore dual-arm OSC:

1. Author `assets/workstations/ar5_l6_bench_bimanual/recipe.yaml`:
   ```yaml
   components:
     base: {component: bases/bench_table, variant: default}
     arm_left:  {component: arms/ar5, variant: left}
     arm_right: {component: arms/ar5, variant: right}
     hand_left:  {component: hands/linkerhand_l6, variant: left}
     hand_right: {component: hands/linkerhand_l6, variant: right}
   freeze_base: base
   mounts:
     - {child: arm_left:base_mount,   parent: base:arm_left_mount}
     - {child: arm_right:base_mount,  parent: base:arm_right_mount}
     - {child: hand_left:wrist_mount, parent: arm_left:tool0,  xyz: [0,0,0.03], rpy: [0,0,1.5708]}
     - {child: hand_right:wrist_mount,parent: arm_right:tool0, xyz: [0,0,0.03], rpy: [0,0,1.5708]}
   ```
2. Extend the manifest schema with `ee_links: dict[role, str]`
   (currently `ee_link: str` picks the first arm — arbitrary for
   bimanual). Both `sim/registry.py` and the composer's
   manifest-writer need updates.
3. Extend `TestOscRLEnv` to instantiate one OSC controller per arm
   role. The env already loops over `_robots`; the refactor is to
   loop over arm *roles* within a single robot.

---

## Known sharp edges

### Gain profiles are hardcoded for OSC
`sim/backends/isaac/loaders.py` applies `kp=150, kd=8` for the `arm`
role when `control_mode="osc"`. Legacy `AR5_L6_*_OSC_CFG` used the
same values. This is a per-robot decision (Franka would want
different OSC gains). Move to `meta.yaml::gain_profiles` in PR #2.

### `sim/envs/test_osc/gain_tuner_osc.py` is unmigrated
Still imports `make_ar5_l6_robot_cfg` and uses `TestOscSceneCfg`
(legacy path). Works because the legacy cfgs remain; gain tuning
isn't affected. Migrate alongside the rest of `sim/envs/test_osc/`
in PR #2.

### MJCF composer is a stub
`tools/composer/mjcf_ops.py::compose_mjcf` raises
`NotImplementedError`. The availability check runs and reports which
component MJCFs are missing. Once all component MJCFs exist, the
check flips green and `compose_mjcf` runs — at which point
`NotImplementedError` fires and we know PR #1b is needed.

### Relative mesh paths escape the workstation directory
Composed URDFs reference meshes via `../../components/…/meshes/<file>`.
This works for local dev and git but requires `tar --dereference`
when packaging a workstation tarball for a real-robot driver. Flag
for a future packaging task; no composer change.

### Phantom-link warning spam on Isaac URDF import

Loading a composed workstation into Isaac Sim 5.1 prints a noisy but
harmless mix of URDF-importer and USD warnings:

- `No mass specified for link <X>` + `Link <X> has no colliders, and no
  inertia was imported; assigning a small isotropic inertia matrix`
- `Unresolved reference prim path @.../configuration/workstation_*.usd@
  </visuals/<X>>` — repeated many times per link because IsaacLab's
  URDF→USD converter stages three sublayers (`workstation_base.usd`,
  `workstation_physics.usd`, `workstation.usd`) and the warning fires
  per recomposition pass × per layer.

Affected links — all composer-introduced and intentionally massless:

- `world` — anchor added by `freeze_base: base`
- `base_workstation_arm_left_mount` / `_right_mount` — intermediate
  frame links for mount joints

Isaac's URDF importer auto-assigns a tiny isotropic inertia, so physics
is correct. The `Unresolved reference` warnings come from empty
`<visual>` groups (the links have no geometry) being re-referenced
across the three sublayers the converter writes.

Clean fix (deferred to PR #2 or whenever it becomes user-visible
friction): in `tools/composer/urdf_ops.py`, emit a stub `<inertial>`
block and either an empty resolvable `<visual>` or no visual at all for
`make_world_link()` and the per-mount intermediate links. ~20 LoC.
Until then: ignore the spam; it does not affect simulation.

Also benign, unrelated to the composer:
- `base_cameraBase / base_cameraJoint: Joint Axis is not body aligned
  with X, Y or Z primary axis` — axis in the `bench_table` component's
  original URDF. PhysX silently reorients. Fix at the base URDF level
  if ever re-authored.

### Hardcoded `_tcp_to_wrist` naming convention
`TestOscRLEnv._tcp_to_wrist` assumes component EE link names end in
`_tcp` and the wrist link differs by suffix `_link7`. Works for AR5;
will break for other arms. Either (a) expose wrist as a second
mount frame in the arm's meta.yaml, or (b) restrict the env's
`ee_frame` choice to `"tcp"` only. Flag for PR #2.

### Workstations can't yet run on MuJoCo
`workstation.mjcf` files aren't generated (PR #1b). MuJoCo-side code
(`sim/backends/mujoco/…`) doesn't exist yet (PR #2). Single-sim
workflow only until both land.

---

## Where to start next

1. If unblocked on MuJoCo authoring: **PR #1b** — hand-author the three
   component MJCFs, flesh out the MJCF composer, add parity checks.
2. If not: **PR #2** — runtime backbone scaffolding (SimBackend,
   Robot protocol, controllers, BaseEnv). Isaac-only until MJCFs
   land; MuJoCo backend and parity tests wait.

Both are independent — they can land in either order or in parallel.
