# Component MJCF authoring guide

This document specifies the contract a component's `<component>.mjcf` file
must satisfy so the workstation composer can merge it with other components
into a single loadable MuJoCo model.

**Scope:** per-component MJCF authoring. The workstation composer handles
name-prefixing, asset deduplication, mount joint insertion, and world
anchoring — authors do not need to anticipate those.

**Why hand-authored and not converted:** URDF→MJCF auto-converters
(`mujoco.MjModel.from_xml_path`, `dm_control.mjcf.from_path`, etc.) produce
models that *load* but have wrong inertias (they run `balanceinertia`
which silently invents mass distributions), wrong collision primitives
(full visual meshes used as colliders), missing sites, and no contact
filtering. A hand-authored MJCF is the single thing that decides whether
MuJoCo sim matches Isaac sim matches real — it has to be curated.

---

## 1. File location

```
assets/components/<kind>/<name>/variants/<variant>/<kind>.mjcf
```

e.g. `assets/components/arms/ar5/variants/left/arm.mjcf`. One MJCF per
variant; they may share geometry via relative `file=` paths into the
variant's `meshes/` directory, but each is a self-contained MuJoCo model.

## 2. Required structure

```xml
<mujoco model="<name>_<variant>">
  <compiler meshdir="meshes" discardvisual="false" autolimits="true"/>
  <option/>                      <!-- left empty; workstation decides integrator/timestep -->

  <default>
    <default class="<kind>_<name>_visual">
      <geom type="mesh" contype="0" conaffinity="0" group="1"/>
    </default>
    <default class="<kind>_<name>_collision">
      <geom type="mesh" contype="1" conaffinity="1" group="3"/>
    </default>
    <!-- add per-joint / per-actuator defaults as needed -->
  </default>

  <asset>
    <mesh name="..." file="..."/>       <!-- one per STL; name them clearly -->
  </asset>

  <worldbody>
    <body name="<root_link from meta.yaml>" pos="0 0 0" quat="1 0 0 0">
      <!-- no freejoint: the workstation composer decides root anchoring -->
      <inertial .../>
      <geom class="..._visual" mesh="..."/>
      <geom class="..._collision" mesh="..."/>
      <site name="<mount_frame_name>" pos="..." quat="..."/>  <!-- one per mount_frame in meta.yaml -->
      <body name="..."> <!-- child body -->
        <joint name="<joint from meta.actuated_joints>" type="hinge" axis="..."
               range="..." armature="..." damping="..." frictionloss="..."/>
        ...
      </body>
    </body>
  </worldbody>

  <actuator>
    <!-- one drive per actuated joint; match meta.actuated_joints order -->
    <position name="<joint>_act" joint="<joint>" kp="..." kv="..." ctrlrange="..."/>
  </actuator>

  <equality>
    <!-- mimic couplings (composer preserves them via name-prefixing) -->
    <joint name="<mimic_name>" joint1="<mimic_joint>" joint2="<driver_joint>"
           polycoef="0 <ratio> 0 0 0"/>
  </equality>

  <contact>
    <!-- optional self-collision exclusions, e.g. between adjacent links -->
    <exclude body1="..." body2="..."/>
  </contact>
</mujoco>
```

## 3. Hard rules (composer enforces these)

1. **Root body matches `meta.yaml`'s `root_link`.** After variable expansion
   (e.g. `{V}` → `L`), the MJCF's top-level `<worldbody>` must contain
   exactly one `<body>` whose `name` matches the component's declared root
   link. The composer attaches this body under the mount parent.

2. **No free joint on the root body.** The workstation composer decides
   whether the assembled workstation is fixed or floating. A component's
   MJCF must not include `<freejoint/>` on its root.

3. **No top-level (unnamed) `<default>`.** Every `<default>` block must
   have a `class="..."` attribute. Unnamed defaults leak into every other
   component's bodies when merged. The composer rejects component MJCFs
   that contain an unnamed default.

4. **`<default>` class names must start with `<kind>_<name>_`** so they
   don't collide across components (e.g. `arm_ar5_visual`, not just
   `visual`). The composer role-prefixes these further at compose time.

5. **Every `mount_frame` in `meta.yaml` must have a matching `<site>`
   inside the component's body tree** (after `{V}` expansion). The
   composer uses the site position to splice mount joints.

6. **Mesh files live under `variants/<v>/meshes/`** and are referenced
   via `file="<basename>.stl"` with `<compiler meshdir="meshes"/>` at
   the top.

7. **Joint names match `meta.yaml`'s `actuated_joints` + `mimic_joints`
   lists exactly** (after `{V}` expansion). Order in MJCF's actuator
   section must match the `actuated_joints` order — the MuJoCo backend
   assumes `data.ctrl` layout follows this order.

## 4. Inertia standards

- **Never use `<compiler balanceinertia="true"/>`.** It silently invents
  mass for bodies whose inertia tensor isn't valid. Fix bad inertias at
  the source (usually a CAD export issue).
- **Source of truth is the URDF.** Inertia values in the component's
  `.mjcf` should match its sibling `.urdf` to within 0.1%. A parity test
  (PR #1b) checks this numerically per-body.
- **Mass must be non-zero** on every dynamic body. Tip-of-finger links in
  the L6 hand URDFs use `mass=1e-6` placeholders; keep the same in MJCF
  (don't drop to zero).

## 5. Collision conventions

- **Separate visual and collision geoms.** `contype=0 conaffinity=0 group=1`
  for visual (rendered, not simulated); `contype=1 conaffinity=1 group=3`
  for collision (simulated, not rendered in default view).
- **Prefer primitive collision geoms** (box, capsule, cylinder, sphere)
  over mesh colliders where possible. Full mesh colliders are slow and
  produce noisy contacts. Swap to convex-decomposed STL only when primitive
  approximation is unacceptable.
- **Collision meshes may differ from visual meshes.** If a component ships
  a `meshes_collision/` subdir, reference those from collision `<geom>`s.
  The current AR5 + L6 components don't have this; full-mesh colliders
  are acceptable as a starting point, tune later.

## 6. Actuator conventions

- **One actuator per entry in `meta.actuated_joints`,** in the same order.
- **Actuator name = `<joint_name>_act`** so the composer can auto-derive
  the joint→actuator mapping when building the manifest.
- **Use `<position>` actuators as the default.** The runtime controller
  (OSC / joint PD) computes torques from position targets; using
  `<motor>` (raw torque) is reserved for controllers that need it.
- **`ctrlrange` matches joint `range`** unless the component deliberately
  restricts it.

## 7. Equality constraints (mimic couplings)

URDF's `<mimic>` and MuJoCo's `<equality polycoef>` describe the same
thing. The component's MJCF should carry the MuJoCo form; the URDF form
lives in the component's URDF. Example: for AR5+L6, the hand's URDF
already has an embedded `<mujoco><equality>` block that specifies the
thumb and finger DIP couplings — that block's contents move into the
hand component's MJCF at the top level `<equality>` section.

Composer name-prefixes the `joint1=` / `joint2=` references automatically;
authors write them with raw (unprefixed) source-names.

## 8. Self-collision policy

- **Do not set `enabled_self_collisions=True` at the component level.**
  The workstation composes many components; self-collision is a scene
  decision, not a component decision.
- **Add `<exclude>` pairs** for geometry pairs that should never collide
  regardless of scene (e.g. a finger's proximal and distal links when the
  collision meshes overlap at the joint).
- **Do not pre-exclude cross-component pairs** (arm last link vs hand
  first link). The composer can emit a mount-seam exclude automatically
  if the recipe requests it.

## 9. What the MJCF validator checks (PR #1b)

For each component variant:

1. `mujoco.MjModel.from_xml_path(<mjcf>)` loads without warnings.
2. Every joint in `meta.actuated_joints` + `meta.mimic_joints` exists in
   the compiled model.
3. Every mount frame in `meta.mount_frames` exists as a `<site>`.
4. Root body name matches `meta.root_link` after `{V}` expansion.
5. No unnamed `<default>` blocks.
6. Inertia tensors match the URDF's within 0.1% per body.
7. Gravity-compensated hold for 500 steps drifts < 1 mrad per joint.

For each composed workstation (PR #1b):

8. MuJoCo loads the composed `workstation.mjcf` without warnings.
9. Joint order in compiled MuJoCo model matches the manifest's
   `joints[role]` list.
10. EE site pose at default pose matches the URDF-derived EE link pose
    within 1e-5 m / 1e-5 rad.

## 10. Checklist for authoring a new component MJCF

- [ ] File at `variants/<v>/<kind>.mjcf`.
- [ ] Root body name matches `meta.root_link` (after `{V}` expansion).
- [ ] No freejoint on root.
- [ ] One `<site>` per entry in `meta.mount_frames`.
- [ ] Joint names match `meta.actuated_joints` + `meta.mimic_joints`.
- [ ] One `<actuator>` per actuated joint, in order, named `<joint>_act`.
- [ ] All `<default>` classes named; prefix starts with `<kind>_<name>_`.
- [ ] Meshes referenced with relative paths, `<compiler meshdir="meshes"/>`.
- [ ] Inertias match sibling URDF to 0.1%.
- [ ] Visual and collision geoms separated via `contype` / `conaffinity`.
- [ ] `balanceinertia` is **not** set.
- [ ] Equality constraints for mimic couplings (if any) present at top level.
- [ ] `mujoco.MjModel.from_xml_path(...)` loads with no warnings.
- [ ] Gravity-compensated hold drifts < 1 mrad / 500 steps.
