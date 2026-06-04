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
  <compiler meshdir="meshes" discardvisual="false" autolimits="true" angle="radian" eulerseq="XYZ"/>
  <option/>                      <!-- left empty; workstation decides integrator/timestep -->
```

**Compiler attribute notes** — the four non-default attributes are *required*, not stylistic:

- `angle="radian"`: URDF rpy is mandated radians, but MuJoCo defaults to degrees for `euler` attributes. Without this, `euler="1.5708 0 0"` becomes a 1.6° rotation instead of 90°.
- `eulerseq="XYZ"` (uppercase): URDF rpy is fixed-axis (extrinsic) XYZ; MuJoCo's lowercase `"xyz"` is intrinsic, which produces a *different* rotation when more than one axis is non-zero. The bug is silent — single-axis rotations agree, only chains with rpy on multiple axes diverge.
- `discardvisual="false"`: keeps visual geoms in the compiled model so authors can verify them in `mujoco.viewer`.
- `autolimits="true"`: applies joint `range` as a limit automatically without needing `limited="true"` per joint.

```xml
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

8. **`<compiler>` must declare `angle="radian"` and `eulerseq="XYZ"`** to
   match the URDF's units and rotation convention. See the discussion in §2.

## 4. Inertia standards

- **Never use `<compiler balanceinertia="true"/>`.** It silently invents
  mass for bodies whose inertia tensor isn't valid. Fix bad inertias at
  the source (usually a CAD export issue). Note the L6 hand URDF's
  embedded `<mujoco>` block sets `balanceinertia="true"` — strip that
  attribute when lifting the block into the component MJCF.
- **Source of truth is the URDF.** Inertia values in the component's
  `.mjcf` should match its sibling `.urdf` to within 0.1%. The validator
  compares **sorted principal-moment eigenvalues** (rotation-invariant)
  rather than the raw tensor, since MuJoCo diagonalizes at compile time.
  This catches axis swaps and signed off-diagonal errors that a trace-only
  check would miss.
- **MJCF inertia format:** `<inertial pos="..." mass="..."
  fullinertia="ixx iyy izz ixy ixz iyz"/>`. Note the order — URDF lists
  inertia attributes as `ixx ixy ixz iyy iyz izz` (interleaved), MJCF
  groups diagonals first then off-diagonals.
- **Mass must be non-zero** on every dynamic body. URDF placeholder links
  with `mass=0` (e.g. `tcp` on AR5) become **welded** MJCF bodies (no
  joint, no inertia, no geom — just a frame for a mount site). The
  validator skips inertia parity for URDF links with `mass < 1e-9`.
  Tip-of-finger links with `mass=1e-6` are above this threshold and
  *are* checked — copy their tiny inertials verbatim from the URDF.

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

**Trim `polycoef` to exactly 5 elements.** The L6 URDF's embedded MJCF
block writes 6 (`"0 1.125676 0 0 0 0"`); MuJoCo expects 5. Drop the
trailing zero.

Composer name-prefixes the `joint1=` / `joint2=` references automatically;
authors write them with raw (unprefixed) source-names.

## 8. Self-collision policy

- **Do not set `enabled_self_collisions=True` at the component level.**
  The workstation composes many components; self-collision is a scene
  decision, not a component decision.
- **Adjacent-link `<exclude>` pairs are required, not optional, when
  using full-mesh colliders.** With shared visual+collision meshes (the
  current AR5 / lkls73 / L6 components), adjacent links interpenetrate at
  the joint axis at qpos=0, and the contact friction directly clamps the
  joint — the actuator can't drive it. Symptom: joint slider moves the
  joint visually but the actuator can't track its target. Cure: add
  `<exclude body1="..." body2="..."/>` between every adjacent body pair.
  The validator's SELF_CONTACT check (§9 #8) flags any active contacts
  at qpos=0 and lists the offending pairs.
- **Add `<exclude>` pairs** for non-adjacent geometry pairs that overlap
  due to short links (e.g. AR5's link5↔link7 — link6 is small enough that
  5 and 7's meshes touch directly), or for geometry pairs that should
  never collide regardless of scene (e.g. a finger's proximal and distal
  links when the collision meshes overlap at the joint).
- **Do not pre-exclude cross-component pairs** (arm last link vs hand
  first link, bench top vs arm link1). The composer auto-emits these
  excludes for every (ancestor, descendant) component pair along the
  mount chain. Mechanism: for each mount, every collision-bearing body
  in the descendant component is excluded against every collision-bearing
  body in each of its mount-ancestors (transitive). For an
  `arm_l6_bench` workstation that means: bench↔arm, bench↔hand, arm↔hand
  excludes are emitted automatically; sibling pairs (`arm_left`↔`arm_right`)
  are *not* excluded so legitimate cross-arm collisions still simulate.
  The workstation validator's `mjcf.self_contact` check fails if any
  active contact remains at qpos=0.

## 9. What the MJCF validator checks

Run `python -m linker_sim.tools.validate_component_mjcf <component_dir> [--variant NAME]`.
The tool implements the following checks per component variant; exits 0
on OK/WARN, 1 on FAIL.

1. **LOAD** — `mujoco.MjModel.from_xml_path(<mjcf>)` loads with no warnings.
2. **JOINTS** — Every joint in `meta.actuated_joints` + `meta.mimic_joints`
   exists in the compiled model.
3. **SITES / SITE_PARENTS** — Every mount frame in `meta.mount_frames`
   exists as a `<site>`, and that site lives inside the body specified by
   `meta.mount_frames[*].parent`.
4. **ROOT** — Body 1 (the first non-world body) matches `meta.root_link`
   after `{V}` expansion.
5. **DEFAULTS** — No unnamed nested `<default>` blocks under the
   top-level `<default>`.
6. **INERTIA_PARITY** — Per-body mass, COM, and **sorted principal-moment
   eigenvalues** match the URDF's `<inertial>` to 0.1% relative tolerance.
   URDF links with `mass < 1e-9` are skipped (placeholder frames).
7. **FK_PARITY** — Forward kinematics at qpos=0: every URDF link's
   world-frame position and rotation match the compiled MJCF body's pose
   to 1e-5. **The strongest correctness check** — catches angle-unit,
   euler-sequence, axis-sign, and origin-transcription bugs that the
   inertia check (which is body-local) won't see.
8. **SELF_CONTACT** — No active contacts at qpos=0 with the default
   collision filter. Failures list the offending body pairs so excludes
   can be added directly.
9. **GRAV_HOLD** — Gravity-compensated, contact-disabled hold for 500
   simulation steps after a 200-step warm-up. Max steady-state joint drift
   must be under 2 mrad. (The original 1 mrad spec is a numerical floor
   not always achievable for mimic-equality chains with tiny armature/inertia
   like the L6 hand; 2 mrad still catches real instability.) Auto-skipped
   for components with no actuated joints.

For each composed workstation (PR #1b — separate validator):

10. MuJoCo loads the composed `workstation.mjcf` without warnings.
11. Joint order in compiled MuJoCo model matches the manifest's
    `joints[role]` list.
12. EE site pose at default pose matches the URDF-derived EE link pose
    within 1e-5 m / 1e-5 rad.

## 10. Checklist for authoring a new component MJCF

- [ ] File at `variants/<v>/<kind>.mjcf`.
- [ ] `<compiler>` declares `angle="radian" eulerseq="XYZ" discardvisual="false" autolimits="true"` and a `meshdir="meshes"`.
- [ ] Root body name matches `meta.root_link` (after `{V}` expansion).
- [ ] No freejoint on root.
- [ ] One `<site>` per entry in `meta.mount_frames`, inside the body matching `parent:`.
- [ ] Joint names match `meta.actuated_joints` + `meta.mimic_joints`.
- [ ] One `<actuator>` per actuated joint, in order, named `<joint>_act`.
- [ ] All `<default>` classes named; prefix starts with `<kind>_<name>_`.
- [ ] Meshes referenced with relative paths.
- [ ] URDF mesh `scale` carried over to `<asset><mesh scale="..."/>` (URDF
      puts scale on every `<visual>`/`<collision>`; MJCF puts it on the
      asset once).
- [ ] Inertias match sibling URDF to 0.1% — use `fullinertia="ixx iyy izz ixy ixz iyz"` (MJCF order, *not* URDF order).
- [ ] Visual and collision geoms separated via `contype` / `conaffinity`.
- [ ] `balanceinertia` is **not** set (strip it if lifting from URDF's embedded `<mujoco>` block).
- [ ] Equality constraints for mimic couplings (if any) present at top level, with `polycoef` of exactly 5 elements.
- [ ] `<contact><exclude>` for every adjacent body pair (full-mesh colliders interpenetrate at joints).
- [ ] `python -m linker_sim.tools.validate_component_mjcf <component_dir>` reports `=> overall: OK`.
- [ ] Visual smoke check in `mujoco.viewer` — drag each joint slider, verify mimic couplings track, verify mount-frame sites land where expected.
