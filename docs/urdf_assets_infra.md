# URDF assets infrastructure

A single-page overview of how robot assets are authored, composed, and
consumed in this repo. Scope is strictly the URDF side of the pipeline —
MJCF authoring is a separate workstream (see
[component_mjcf_authoring.md](component_mjcf_authoring.md)).

---

## 1. What problem this solves

A "workstation" is the full kinematic thing we spawn in Isaac: base +
one or more arms + one or more hands, sometimes with sensors, welded
into a single articulated URDF the simulator can load.

Authoring these URDFs by hand for every base/arm/hand combination
scales poorly:
- N bases × M arms × K hands = N·M·K URDFs to maintain.
- Link/joint names collide across copies of the same arm (left/right).
- Mesh filenames break when files move.
- Gain tuning drifts between copies of the "same" arm.

Instead we keep **components** (self-contained subtrees: one base, one
arm variant, one hand variant) and let a **composer** stitch them into
a **workstation** by following a declarative **recipe**. The composer
emits a flat URDF plus a manifest that the runtime reads to know what
it's driving.

---

## 2. Directory layout

```
assets/                                  # repo-root for now; Phase 2 moves into linker-robot-assets
  components/
    arms/
      ar5/
        meta.yaml
        variants/left/{arm.urdf, meshes/*.STL}
        variants/right/{arm.urdf, meshes/*.STL}
      lkls73_arm/ ...
    bases/
      bench_table/  lkls73_torso/ ...
    hands/
      linkerhand_l6/ ...
  workstations/
    ar5_l6_bench_bimanual/
      recipe.yaml          # authored
      workstation.urdf     # generated, committed
      manifest.yaml        # generated, committed
    lkls73_i1_bimanual/
packages/linker-sim/src/linker_sim/
  tools/
    composer/{compose.py, urdf_ops.py, schemas.py, determinism.py, mjcf_ops.py}
    validate_workstation.py
    registry_show.py
    ci/check_drift.sh
  registry.py              # runtime entry point
```

Components are the unit of reuse. Workstations are the unit of
deployment — one per physical robot configuration we care about.

---

## 3. The three YAML contracts

### 3.1 Component `meta.yaml`

Declares what a component exposes to the composer. Authored by hand.
Schema lives in [linker_sim/tools/composer/schemas.py:112-205](../packages/linker-sim/src/linker_sim/tools/composer/schemas.py#L112-L205).

Key fields:
- `kind`: `arm | hand | base | sensor`
- `variants`: named variants (e.g. `left` / `right`) each pointing at a
  URDF and a mesh dir. A variant carries `vars: {V: L}` — these expand
  `{V}` placeholders in *meta fields only*, not in the URDF body.
- `root_link`: the subtree's root link (may contain `{V}`).
- `mount_frames`: named points where another component can attach.
  Each entry names a parent link; attachment point defaults to that
  link's origin unless `xyz`/`rpy` are set.
- `ee_frame` (arms only): which mount frame serves as the end-effector.
- `actuated_joints` / `mimic_joints`: declared-contract joint names,
  for validation against the URDF's actual joints.
- `default_gains` + `gain_profiles` (e.g. `joint`, `osc`): controllers
  pick a profile at attach time, so the same arm can run direct
  joint-space PD or OSC with different PD gains without reloading.

### 3.2 Workstation `recipe.yaml`

The composition spec. Schema at
[linker_sim/tools/composer/schemas.py:265-340](../packages/linker-sim/src/linker_sim/tools/composer/schemas.py#L265-L340).

```yaml
components:
  base:       { component: bases/bench_table,    variant: default }
  arm_left:   { component: arms/lkls73_arm,      variant: left    }
  arm_right:  { component: arms/lkls73_arm,      variant: right   }
  hand_left:  { component: hands/linkerhand_l6,  variant: left    }
  hand_right: { component: hands/linkerhand_l6,  variant: right   }
freeze_base: base
mounts:
  - { child: arm_left:base_mount,  parent: base:arm_left_mount,  xyz: [0,0,0], rpy: [0,0,0] }
  - { child: arm_right:base_mount, parent: base:arm_right_mount, xyz: [0,0,0], rpy: [0,0,0] }
  - { child: hand_left:wrist_mount,  parent: arm_left:tool0,  xyz: [0,0,0], rpy: [pi,0,0] }
  - { child: hand_right:wrist_mount, parent: arm_right:tool0, xyz: [0,0,0], rpy: [pi,0,0] }
physics_overrides: {}
```

- **Roles** (`base`, `arm_left`, …) are arbitrary labels; they become
  link/joint name prefixes and are the addressing key at runtime
  (`handle.joints["arm_left"]`, `handle.ee_links["arm_right"]`).
- **`freeze_base: <role>`** adds a `world` link and a fixed joint from
  world to the named component's `root_link`.
- **`mounts`** are **always fixed joints** — this is a hard composer
  invariant ([urdf_ops.py:182-194](../packages/linker-sim/src/linker_sim/tools/composer/urdf_ops.py#L182-L194)).
  See §6 for the consequence.

### 3.3 Generated `manifest.yaml`

Written by the composer, committed to git. Schema at
[schemas.py:373-430](../packages/linker-sim/src/linker_sim/tools/composer/schemas.py#L373-L430). Contains:
- sha256 of recipe + every component's sources
- sha256 of the composed URDF
- per-role actuated/mimic joint lists (prefixed names)
- `ee_link`, `ee_links[role]`, `base_link`, `frames`
- merged `default_gains` and `gain_profiles` per role
- `components` provenance: which component directory + variant + hash

The manifest is the source of truth at runtime — the registry never
re-parses the URDF.

---

## 4. Composition algorithm

Entry point: `python -m linker_sim.tools.composer.compose <workstation_dir>`
([linker_sim/tools/composer/compose.py:130-291](../packages/linker-sim/src/linker_sim/tools/composer/compose.py#L130-L291)).
Pipeline:

1. **Load** the recipe and each referenced component's `meta.yaml`.
2. **Resolve variants** — pick the named variant (or the only one).
3. **Per component, compile**
   ([urdf_ops.py:222-257](../packages/linker-sim/src/linker_sim/tools/composer/urdf_ops.py#L222-L257)):
   - parse the variant's URDF
   - prefix every `link/joint/material/transmission name` with
     `<role>_`; rewrite every `<parent link>`, `<child link>`,
     `<mimic joint>`, and `<mujoco><equality joint1/joint2>` reference
   - rewrite `<mesh filename>` paths so they resolve from the
     workstation directory (meshes stay in place; no copies)
   - collect actuated vs mimic joint names in document order
4. **Merge** ([urdf_ops.py:263-416](../packages/linker-sim/src/linker_sim/tools/composer/urdf_ops.py#L263-L416))
   into one `<robot>`:
   - optional `world` link (if `freeze_base` is set)
   - deduped top-level `<material>` defs
   - links, then joints, then transmissions (component document order,
     then role order — deterministic)
   - concatenated `<mujoco>` blocks with a single compatible `<compiler>`
   - **mount joints** (fixed, in recipe order), named
     `mount_<child_role>_to_<parent_role>_<frame>`
   - optional world → base fixed joint
5. **Hash** the serialized URDF; build the manifest.
6. **Write if changed** (`workstation.urdf`, `manifest.yaml`, and
   `workstation.mjcf` when all component MJCFs are present; MJCF
   emission is stubbed pending PR #1b).

Determinism is enforced by
[linker_sim/tools/composer/determinism.py](../packages/linker-sim/src/linker_sim/tools/composer/determinism.py) —
stable float formatting + consistent indentation — so composed output
is byte-stable across runs and platforms.

---

## 5. Runtime consumption

One file: [sim/registry.py](../sim/registry.py).

```python
from sim.registry import discover, load

names = discover()                  # ["ar5_l6_bench_bimanual", "lkls73_i1_bimanual"]
handle = load("lkls73_i1_bimanual")

handle.urdf_path                    # absolute Path, ready for Isaac
handle.joints["arm_left"]           # 7 prefixed joint names
handle.ee_links["arm_right"]        # prefixed link name for OSC target
handle.gain_profiles["arm_left"]["osc"]   # Gains(stiffness=150, damping=8)
handle.joint_names()                # flat list across all roles
handle.role_of(joint_name)          # reverse lookup
```

`WorkstationHandle` is intentionally thin — it's a typed read of the
manifest plus absolute paths. The registry never re-parses URDFs and
never talks to a simulator. All sim-specific logic
(`to_articulation_cfg`, collider settings, etc.) lives in
`sim/backends/<backend>/`.

Roles are the addressing contract. Bimanual code asks for
`handle.joints["arm_left"]` and `handle.ee_links["arm_right"]`;
controllers and tasks are role-parametric and pick up any new
workstation that declares the expected roles.

---

## 6. Key invariants (and one gotcha)

### Mount joints are always fixed

The composer only emits `type="fixed"` for recipe mounts. The first
actuated joint of an arm must therefore live **inside** the arm
component's URDF — you can't lift it into the recipe.

Two patterns handle this:
- **Real base link exists in the source** (AR5): the arm URDF is
  already rooted at its own stationary "base" link; joint_1 is internal.
- **No intermediate link in the source** (lkls73): the arm variant URDF
  carries a zero-mass virtual root (`L_arm_root` / `R_arm_root`); the
  first revolute (`L1_Joint`) hangs from it with identity origin. The
  positional offset the source URDF had on `L1_Joint` is absorbed into
  the parent base's mount-pad link instead. See
  [lkls73_torso/base.urdf:42-54](../assets/components/bases/lkls73_torso/variants/default/base.urdf#L42-L54)
  and [lkls73_arm/variants/left/arm.urdf](../assets/components/arms/lkls73_arm/variants/left/arm.urdf).

### `{V}` expansion applies to meta fields only

`root_link`, `mount_frames.parent`, and `actuated_joints` may contain
`{V}`. The *component URDF body* is NOT substituted — variant URDFs
ship with literal `L…`/`R…` names. This is why the AR5 and lkls73 arms
each have two separate URDFs per hand.

### Meshes are referenced, not copied

`rewrite_mesh_paths` only edits the `<mesh filename>` attribute to be a
path relative to the *workstation* directory. Mesh files never leave
their component directory. Consequence: moving a component breaks every
workstation that references it until you re-compose.

### Generated artifacts are committed

`workstation.urdf` and `manifest.yaml` live in git. Rationale:
reviewers can diff them, CI doesn't need to run the composer to load a
workstation, and the manifest's sha fields catch stale artifacts
automatically.

---

## 7. Validation & CI

Three complementary gates:

| Tool | What it checks |
|---|---|
| `python -m linker_sim.tools.composer.compose <ws>` | Compose cleanly; errors on schema problems, mesh resolution, unknown mount frames. |
| `python -m linker_sim.tools.validate_workstation <ws>` | 8 checks: manifest hash self-consistency (recipe, components, URDF), joint-count vs URDF, EE/mount link resolution, mesh files on disk, single connected kinematic tree, drift ([linker_sim/tools/validate_workstation.py:78-226](../packages/linker-sim/src/linker_sim/tools/validate_workstation.py#L78-L226)). |
| `bash packages/linker-sim/src/linker_sim/tools/ci/check_drift.sh` | Re-runs the composer in memory for every workstation; fails if any committed artifact diverges from fresh output. CI uses this. |

Inspection: `python -m linker_sim.tools.registry_show <ws>` prints the loaded
`WorkstationHandle` — roles, joints, frames, gains — without involving
any sim backend.

Recommended workflow when editing a component or recipe:

```bash
python -m linker_sim.tools.composer.compose assets/workstations/<ws>       # regen
python -m linker_sim.tools.validate_workstation assets/workstations/<ws>   # sanity
bash packages/linker-sim/src/linker_sim/tools/ci/check_drift.sh                                    # all green
git add assets/workstations/<ws>/{workstation.urdf,manifest.yaml}
```

---

## 8. Extending the asset library

### Adding a new variant to an existing component

1. Author `variants/<new>/arm.urdf` with literal names for that variant.
2. Add an entry to `variants:` in the component's `meta.yaml`.
3. Reference it from a recipe. Compose, validate, commit.

### Adding a new component

1. Pick `kind` (arm/hand/base/sensor) and create
   `assets/components/<kind>/<name>/`.
2. Write `meta.yaml` (declare `root_link`, `mount_frames`, `ee_frame`
   for arms, `actuated_joints`, gain profiles).
3. Drop the variant URDF(s) and mesh dir(s). If the source URDF's first
   joint is revolute (no intermediate base link), insert a zero-mass
   virtual root link so the recipe's fixed mount can land on it.
4. Build a minimal test recipe, compose, validate.

### Adding a new workstation

1. `assets/workstations/<name>/recipe.yaml` with roles, mounts, and
   `freeze_base`.
2. Compose → validate → drift. Commit the three files.
3. Add a Hydra config under `sim/configs/robot/<name>.yaml`:

   ```yaml
   # @package _global_
   robot:
     workstation_name: <name>
     role_name: robot
     rigid_bodies: {}
   ```
4. Run headless smoke via `scripts/run.py`.

---

## 9. Reference paths

- Composer entry: [linker_sim/tools/composer/compose.py](../packages/linker-sim/src/linker_sim/tools/composer/compose.py)
- Composer primitives: [linker_sim/tools/composer/urdf_ops.py](../packages/linker-sim/src/linker_sim/tools/composer/urdf_ops.py)
- Schemas: [linker_sim/tools/composer/schemas.py](../packages/linker-sim/src/linker_sim/tools/composer/schemas.py)
- Determinism: [linker_sim/tools/composer/determinism.py](../packages/linker-sim/src/linker_sim/tools/composer/determinism.py)
- Validator: [linker_sim/tools/validate_workstation.py](../packages/linker-sim/src/linker_sim/tools/validate_workstation.py)
- CI drift gate: [linker_sim/tools/ci/check_drift.sh](../packages/linker-sim/src/linker_sim/tools/ci/check_drift.sh)
- Runtime registry: [sim/registry.py](../sim/registry.py)
- Inspection CLI: [linker_sim/tools/registry_show.py](../packages/linker-sim/src/linker_sim/tools/registry_show.py)
- Example component (arm): [assets/components/arms/ar5/meta.yaml](../assets/components/arms/ar5/meta.yaml)
- Example workstation (bimanual humanoid): [assets/workstations/lkls73_i1_bimanual/recipe.yaml](../assets/workstations/lkls73_i1_bimanual/recipe.yaml)
