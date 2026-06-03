# PR #3 progress log

**Status:** landed and **end-to-end verified** (2026-05-11). Bimanual
workstation composes, validates, drift-free, spawns in Isaac, and
rolls out under OSC with both arms visibly moving through the action
layout `[arm_left(6), hand_left(6), arm_right(6), hand_right(6)]`
(`action_dim=24`). Tier 4 of `docs/TEST_PIPELINE.md` is the gate and
it's green.

**Audience:** future readers picking up handover / bimanual pick-place
tasks, or anyone extending the composer to more than two arm roles.

---

## What PR #3 is

A composed `ar5_l6_bench_bimanual` workstation + the small manifest
and task plumbing needed to drive it. The runtime backbone from PR #2
didn't need structural changes — everything here lands in recipes,
the composer, the registry, and one new task class.

### Delivered

- **Bimanual recipe**:
  `assets/workstations/ar5_l6_bench_bimanual/recipe.yaml` — 5 roles
  (base, arm_left, arm_right, hand_left, hand_right), 4 mounts, one
  composed URDF.
- **Per-role ee_links** on `Manifest` + `WorkstationHandle`. Single
  `ee_link` stays as the first-arm alias for back-compat (existing
  code that only reads `handle.ee_link` keeps working).
- **Composer fix for duplicate `<mujoco><compiler>` blocks**: two
  hand components now share a compiler element — composer accepts
  identical compilers and drops duplicates; raises on mismatched
  compilers. (Previously the bimanual recipe would hit a `SchemaError`
  because the old check refused any multi-compiler scene.)
- **`BimanualReachTask`** — two independent targets, summed reward,
  termination requires both arms in-threshold for `success_hold_steps`.
- **Hydra configs**: `robot/ar5_l6_bench_bimanual.yaml`,
  `controller/osc_bimanual.yaml`, `task/bimanual_reach.yaml`.
- **Backend comment refresh**: the `len(cfg.workstations) != 1` gate
  now explains the distinction between "multi-role single articulation"
  (supported) and "multi-articulation scenes" (not supported).
- **Tests**: `test_bimanual_manifest.py` (6 tests) and
  `test_bimanual_reach_task.py` (4 tests), pure-Python, no Isaac.

---

## Design decisions

**D19 (single articulation, multiple roles).** The bimanual
workstation is ONE composed URDF → ONE articulation in Isaac. The
backend still exposes `backend.robots = {"robot": IsaacRobot(...)}`;
controllers differentiate via `cfg.role="arm_left"` and route through
`robot.joint_ids_of(role)`. Multi-articulation scenes (genuinely
separate robots sharing an env) are not supported and would require
a different `InteractiveScene` layout.

**D20 (per-role ee_links).** The manifest had one `ee_link` (first arm
wins). Bimanual needs two. Added `ee_links: dict[str, str]` keyed by
role; retained `ee_link` as a back-compat alias. `IsaacRobot._resolve_frame`
already looks up `"role:frame_name"` via `handle.frames` — no `Robot`
API change needed. Tasks pass `ee_frame="arm_left:tool0"` / `"arm_right:tool0"`.

**D21 (bimanual reach = two independent targets).** Reward sums
per-arm shaped distance/orientation terms; termination requires both
arms' success streaks. Action layout:
`[arm_left(6), hand_left(6), arm_right(6), hand_right(6)]` — 24 dims.

---

## File map

### New
- `assets/workstations/ar5_l6_bench_bimanual/` — recipe + composed
  `workstation.urdf` + `manifest.yaml`.
- `sim/tasks/bimanual_reach.py` — task + `ArmSpec` + `BimanualReachTaskCfg`.
- `sim/configs/robot/ar5_l6_bench_bimanual.yaml`
- `sim/configs/controller/osc_bimanual.yaml`
- `sim/configs/task/bimanual_reach.yaml`
- `tests/test_bimanual_manifest.py`
- `tests/test_bimanual_reach_task.py`

### Edited
- `tools/composer/schemas.py` — `Manifest.ee_links: dict[str, str]`.
- `tools/composer/compose.py` — populates `ee_links` from every
  component that declares `ee_frame`.
- `tools/composer/urdf_ops.py` — relaxed the multi-compiler check:
  identical `<mujoco><compiler>` attributes are accepted; subsequent
  duplicates are stripped.
- `sim/registry.py` — `WorkstationHandle.ee_links: dict[str, str]`.
- `sim/backends/isaac/backend.py` — comment refresh on the
  single-articulation gate.
- `assets/workstations/ar5_l6_bench/manifest.yaml` and `_right` —
  re-composed to pick up the new `ee_links` field (one entry each).
- `tests/test_base_env.py` — `_make_handle` now passes `ee_links`.
- `docs/PR2_PROGRESS.md` — pointer to this doc.

---

## Dev workflow

```bash
# Compose the bimanual workstation (URDF + manifest).
python -m tools.composer.compose assets/workstations/ar5_l6_bench_bimanual

# Validate.
python tools/validate_workstation.py assets/workstations/ar5_l6_bench_bimanual

# Full CI drift check (all three workstations).
bash tools/ci/check_drift.sh

# Unit tests (no Isaac).
pytest tests/test_bimanual_manifest.py tests/test_bimanual_reach_task.py

# Visual smoke: both arms reach toward sampled targets.
python scripts/run.py \
    robot=ar5_l6_bench_bimanual \
    controller=osc_bimanual \
    task=bimanual_reach
```

---

## Known sharp edges

- **Motion rollout verified** — both arms move, both hands move,
  `action_dim=24`, long-horizon stability holds. Tier 4 of the test
  pipeline cleared on 2026-05-11.
- **Phantom-link warnings** — bimanual adds two more mounts, so the
  warning count roughly doubles. Root cause unchanged (composer
  doesn't know which pre-base links are unused). Still a queued
  fix, ~20 LoC.
- **MuJoCo real impl** — still PR #1b-blocked. The bimanual URDF
  also carries two `<mujoco><equality>` blocks (one per hand) — the
  composer currently concatenates them, which MuJoCo accepts; PR #1b
  should verify this when MJCF authoring lands.
- **Bimanual pick_place / handover** — deferred. The existing
  pick_place task already takes `robot_name` / `ee_frame`, so
  wiring it to one arm of the bimanual works today; what's missing
  is the cross-arm grasp-transfer reward shaping.
- **Real-robot driver** — the composed bimanual URDF is tooling-ready
  but no real-robot backend exists yet. Separate PR.

---

## Where to start next

1. **Actual motion rollout** — drive
   `scripts/run.py robot=ar5_l6_bench_bimanual controller=osc_bimanual task=bimanual_reach policy=random_walk`
   and confirm both arms visibly respond to commands and close
   distance to their sampled targets over an episode. Tune action
   scales / reward weights if convergence is slow.
2. **PR #1b** — component MJCF authoring. Unblocks the MuJoCo backend
   stubs and lets us run the bimanual workstation in MuJoCo too.
3. **Bimanual handover / pick_place** — builds on the existing
   `PickPlaceTask` + a second pluggable arm role.
4. **Agent training** — wire an SB3 / skrl / CleanRL agent around
   `BaseEnv(... task=BimanualReachTask)` as a testbed for bimanual
   policies.
