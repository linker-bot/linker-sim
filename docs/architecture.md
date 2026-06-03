# Architecture

High-level model of the repo for new contributors. The full historical
context behind each design decision lives in the original PR notes,
which are not part of the public release.

## Three-layer model

1. **Asset layer** — `assets/`. Per-component `meta.yaml` and per-side
   variants (URDF + MJCF + meshes) are composed into a per-workstation
   monolithic `workstation.urdf` + `workstation.mjcf` + `manifest.yaml`
   via `tools/composer/`. Generated artifacts are committed; CI gates
   on drift between recipe and committed output.

2. **Runtime layer** — `sim/`. A thin sim-agnostic registry
   (`sim/registry.py`) returns a `WorkstationHandle` that backends
   (`sim/backends/{isaac,mujoco}/`) consume. Controllers
   (`sim/controllers/{joint_pd,osc,ik}.py`) and tasks (`sim/tasks/`)
   are protocol-shaped and backend-agnostic. The manifest is the
   runtime contract — joints, end-effector links, gain profiles.

3. **Apps layer** — `scripts/`. Hydra entrypoints (`run.py`, `replay.py`,
   `replay_ik.py`) glue runtime + tasks + recorders + controllers into
   end-to-end sessions.

## Design decisions worth knowing

- **Author-time composition.** Composition runs once at author time and
  emits monolithic URDF/MJCF. The simulator never splices at load.
  Reason: MuJoCo needs one model root with a flat namespace and a
  consistent meshdir; runtime weld constraints degrade physics fidelity
  and break controllers that need a coherent jacobian from base to
  fingertip.

- **Manifest as runtime contract.** Backends never re-parse URDF/MJCF.
  They read `manifest.yaml` for joint names, end-effector links, gains,
  base/EE frames. Schema changes to the manifest are breaking changes
  for every consumer (sim runtime, real-robot replay, teleop) — see
  `docs/REFACTOR_PLAN.md` for the planned versioning gate.

- **No xacro in the pipeline.** Author-side xacro is fine for component
  authoring, but the composer only consumes flat URDFs. Reason: keep
  the composer dependency surface small and the diff between authored
  and composed XML easy to reason about.

- **Single articulation per workstation, multiple roles.** A bimanual
  workstation is ONE composed URDF → ONE articulation. Backends expose
  `backend.robots = {"robot": ...}`; controllers dispatch by role
  (`arm_left`, `arm_right`, `hand_left`, `hand_right`). Multi-articulation
  scenes (genuinely separate robots sharing one env) are not supported
  and would require a different `InteractiveScene` layout.

- **Per-role end-effector links.** `Manifest.ee_links: dict[str, str]`
  is keyed by role; `ee_link` (singular) is retained as a back-compat
  alias for the first arm.

## UMI-Dex bag → sim replay

The replay pipeline takes UMI-Dex tracker bags (`/vut/pose`,
`/hand/usart_raw`) and produces a `(T, 7)` `[x,y,z,qw,qx,qy,qz]`
trajectory in workstation-base frame plus `(T, n_joints)` hand angles,
serialized as `umi_ee_poses.npz`. `scripts/replay_ik.py` then drives
the workstation through DLS IK in MuJoCo.

Two unknowns are resolved at preprocess time:

- **Map origin** — the VIVE Ultimate Tracker SLAM origin is
  session-arbitrary. Anchored by aligning frame 0 of the bag to the
  robot's default-pose `tool0` FK; the absolute origin then drops out.
- **Anchor refinement** — `scripts/anchor_search.py` runs Nelder-Mead
  over `(dx, dy, dz, anchor_rpy)` to minimize tracking residual on top
  of the centroid anchor.

The rig offset `T_tracker_body←gripper_tcp` is treated as identity for
v0; refinement deferred until orientation error visibly matters.

The hand-angle decoder currently delegates to UMI-Dex's `Calibrator`.
Replacing it with a linker-sim native, manifest-driven decoder is on
the refactor roadmap (see `docs/REFACTOR_PLAN.md` Phase 4).

## Where to read what

| Topic | Doc |
|---|---|
| Install + env setup | [docs/installation.md](installation.md) |
| End-user usage | [docs/USAGE.md](USAGE.md) (English) / [docs/USAGE.zh.md](USAGE.zh.md) (中文) |
| Asset / URDF authoring | [docs/urdf_assets_infra.md](urdf_assets_infra.md) |
| Component MJCF authoring | [docs/component_mjcf_authoring.md](component_mjcf_authoring.md) |
| Test pipeline | [docs/TEST_PIPELINE.md](TEST_PIPELINE.md) |
| Refactor / open-source plan | [docs/REFACTOR_PLAN.md](REFACTOR_PLAN.md) |
