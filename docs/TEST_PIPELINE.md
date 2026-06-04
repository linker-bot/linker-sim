# Isaac Sim pipeline verification

Tiered test pipeline that verifies the work from PR #1 (composer), PR #2
(runtime backbone), and PR #3 (bimanual workstation) end-to-end. Each
tier only runs if the previous tier passed — the earlier a regression
bites, the cheaper it is to diagnose.

Run everything from an activated env:

```bash
source ~/opt/IsaacLab/env_isaaclab/bin/activate
cd <path-to-linker-sim>
```

**Shell hygiene:** if ROS 2 Jazzy is sourced in your default shell
(`setup.fish` / `setup.bash` in rc files), `/opt/ros/jazzy/...` leaks
into `PYTHONPATH` and breaks pytest ("No module named 'lark'", from
`launch_pytest`). Either unsource ROS or strip the env vars per-command:

```fish
env -u PYTHONPATH -u AMENT_PREFIX_PATH <command>
```

---

## Status tracker

Update per tier as it's run. `last-run` uses ISO date (`YYYY-MM-DD`).
`state` is one of: `pending`, `running`, `pass`, `fail`, `blocked`.

| Tier | Scope | State | Last run | Notes |
|------|-------|-------|----------|-------|
| 0    | Pure-Python pytest                    | pass    | 2026-05-11 | All 25 tests green after stripping ROS envs. |
| 1A   | Recompose all workstations            | pass    | 2026-05-11 | 3 workstations recomposed, no errors. |
| 1B   | Validate all workstations             | pass    | 2026-05-11 | All 8 checks green per workstation. |
| 1C   | Drift gate (`check_drift.sh`)         | pass    | 2026-05-11 | No drift across all 3 workstations. |
| 1D   | Registry inspection                   | pass    | 2026-05-11 | Bimanual contract verified (5 roles, distinct ee_links, frames). |
| 2A   | Headless spawn — bimanual default     | pass    | 2026-05-19 | `setup complete for workstation 'ar5_o6_bench_bimanual'`, `action_dim=24`. |
| 2B   | Headless spawn — lkls73_i1_o6         | pending | -          | |
| 2C   | Headless spawn — a7_lite_o6_dc        | pending | -          | |
| 2D   | Headless spawn — ar5_l6 (legacy)      | pending | -          | |
| 2E   | Headless spawn — multi-env (B=16)     | pending | -          | Re-run after default switch. |
| 3A   | Reach + OSC + zeros (single-arm)      | pass    | 2026-05-11 | Arm holds default pose; `action_dim=12`. |
| 3B   | Reach + OSC + random_walk             | pass    | 2026-05-11 | |
| 3C   | Reach + JointPD + zeros               | pass    | 2026-05-11 | Needed `task.cfg.action_dim=13` override (AR5 7 + hand 6). |
| 3D   | Pick-place + OSC + zeros              | pass    | 2026-05-11 | Fixed cube spawn Z (1.29 m bench top) + convex_decomposition collider. |
| 3E   | Right workstation + OSC + random_walk | pass    | 2026-05-11 | |
| 4A   | Bimanual reach + OSC + zeros          | pass    | 2026-05-11 | **PR #3 motion gate cleared — `action_dim=24` confirmed.** |
| 4B   | Bimanual reach + OSC + random_walk    | pass    | 2026-05-11 | Both arms and both hands visibly move. |
| 4C   | Bimanual reach + reward trend         | pass    | 2026-05-11 | JSONL rewards finite, non-NaN, trending as expected. |
| 4D   | Bimanual multi-env (B=4)              | pass    | 2026-05-11 | |
| 5A   | Record a reach episode                | pass    | 2026-05-11 | JSONL episodes contain obs/action/reward/terminated/truncated. |
| 5B   | Action-replay                         | pass    | 2026-05-11 | Doc had `mode=` — needs `+mode=` (hydra struct). Patched. |
| 5C   | State-inject replay                   | pass    | 2026-05-11 | |
| 6A   | Gain tuner hot-reload                 | pass    | 2026-05-11 | Optional; legacy path. |
| 6B   | Long-horizon bimanual stability       | pass    | 2026-05-11 | GPU memory flat over the run. |

---

## Tier 0 — Pure-Python gate (<30s, no GPU)

Verifies the runtime backbone's non-sim invariants. Guards the contracts
between composer, registry, controllers, tasks, and `BaseEnv`. All of it
must be green before any Isaac test is attempted.

```fish
env -u PYTHONPATH -u AMENT_PREFIX_PATH pytest tests/ -v
```

**Expected:** 25 tests pass across `test_base_env.py`, `test_protocols.py`,
`test_reach_task.py`, `test_bimanual_reach_task.py`,
`test_bimanual_manifest.py`, `test_recorder_roundtrip.py`.

**Failure modes:**
- `ModuleNotFoundError: No module named 'lark'` (from `launch_pytest`) →
  ROS 2 leak. See shell hygiene note at the top.
- Registry/manifest failures → schema drift. Run Tier 1A.
- Task-dim failures → controller `command_dim` sum drifted from the
  task's `action_dim`. Cross-check `sim/configs/task/*.yaml` against
  `sim/configs/controller/*.yaml`.

---

## Tier 1 — Composer / validator / drift (<10s, no GPU)

Verifies composed artifacts are byte-identical to what the composer
produces from current sources.

### 1A. Recompose all workstations

```fish
for ws in assets/workstations/*/; python -m linker_robot_assets.composer.compose $ws; end
```

**Expected:** 6 workstations recomposed (`ar5_l6_bench_bimanual`,
`ar5_o6_bench_bimanual`, `lkls73_i1_bimanual`, `lkls73_i1_o6_bimanual`,
`a7_lite_dc`, `a7_lite_o6_dc`). Phantom-link warnings are benign
(1 per `world` + 2 per mount link — see
[PR1_PROGRESS.md:393-426](PR1_PROGRESS.md#L393-L426)).

### 1B. Validate (8 checks: hashes, joint count, EE link, mount frames, mesh paths, single tree, drift)

```fish
for ws in assets/workstations/*/; python -m linker_robot_assets.validate_workstation $ws; end
```

**Expected:** All 8 checks pass per workstation, exit 0.

### 1C. CI drift gate

```fish
bash packages/linker-robot-assets/src/linker_robot_assets/ci/check_drift.sh
```

**Expected:** green across all 6 workstations.

### 1D. Registry inspection (bimanual contract check)

```fish
python -m linker_sim.tools.registry_show                       # lists 3 workstations
python -m linker_sim.tools.registry_show ar5_l6_bench_bimanual # bimanual handle dump
```

**Expected for bimanual:**
- `joints` has 5 role keys: `base`, `arm_left`, `arm_right`, `hand_left`, `hand_right`
- `len(joints["arm_left"]) == 7`, `len(joints["hand_left"]) == 6`, same for right
- `ee_links` contains **distinct** prefixed links for `arm_left` and `arm_right`
- `frames` contains both `arm_left:tool0` and `arm_right:tool0`
- `gain_profiles["arm_left"]["osc"]` and `..["arm_right"]["osc"]` both present, stiffness=150

---

## Tier 2 — Isaac headless scene spawn (~30s each, GPU required)

Verifies composed workstations load into Isaac's URDF importer, spawn as
valid articulations, and run physics without crashing.

**Canonical form** (bounded, line-buffered, tee'd — avoids the
headless+no-max-steps+`| tail` hang trap):

```fish
timeout -k 5 20 env PYTHONUNBUFFERED=1 python scripts/run.py \
    robot=<name> headless=true max_steps=200 policy=zeros 2>&1 | tee /tmp/t2.log
grep -E "backend ready|env ready|Error|Traceback" /tmp/t2.log
```

**Why `-k 5`:** Isaac's `SimulationApp` (kit) ignores plain SIGTERM
from `timeout`; the process will outlive the nominal 20s window. The
`-k 5` flag follows up with SIGKILL 5s later, which kit cannot ignore.
Without `-k`, expect to Ctrl-C the process yourself — the grep
output is still the real pass/fail signal either way.

**Pass criteria:** `grep` output contains `backend ready` and `env ready`,
followed by `done after <N> steps`, with zero `Traceback`/`Error` lines
other than known benign Isaac warnings (phantom-link inertia
auto-assign, `Unresolved reference prim path`).

> Note: tiers 2A/2B previously covered single-arm workstations.
> Unimanual workstations were retired in commit `f1a10f8`; the
> shipped workstations now are bimanual (`ar5_l6_bench_bimanual`,
> `ar5_o6_bench_bimanual`, `lkls73_i1_bimanual`, `lkls73_i1_o6_bimanual`,
> `a7_lite_dc`, `a7_lite_o6_dc`). 2A is the **default** workstation
> (`ar5_o6_bench_bimanual`); 2B–2D exercise alternates and 2E is the
> multi-env stress check.

### 2A. ar5_o6_bench_bimanual (default)

```fish
timeout -k 5 20 env PYTHONUNBUFFERED=1 python scripts/run.py \
    robot=ar5_o6_bench_bimanual headless=true max_steps=200 policy=zeros 2>&1 | tee /tmp/t2a.log
grep -E "backend ready|env ready|done after|Error|Traceback" /tmp/t2a.log
```

### 2B. lkls73_i1_o6_bimanual

```fish
timeout -k 5 20 env PYTHONUNBUFFERED=1 python scripts/run.py \
    robot=lkls73_i1_o6_bimanual headless=true max_steps=200 policy=zeros 2>&1 | tee /tmp/t2b.log
grep -E "backend ready|env ready|done after|Error|Traceback" /tmp/t2b.log
```

### 2C. a7_lite_o6_dc

```fish
timeout -k 5 20 env PYTHONUNBUFFERED=1 python scripts/run.py \
    robot=a7_lite_o6_dc headless=true max_steps=200 policy=zeros 2>&1 | tee /tmp/t2c.log
grep -E "backend ready|env ready|done after|Error|Traceback" /tmp/t2c.log
```

### 2D. ar5_l6_bench_bimanual (legacy L6 hand)

```fish
timeout -k 5 20 env PYTHONUNBUFFERED=1 python scripts/run.py \
    robot=ar5_l6_bench_bimanual headless=true max_steps=200 policy=zeros 2>&1 | tee /tmp/t2d.log
grep -E "backend ready|env ready|done after|Error|Traceback" /tmp/t2d.log
```

`backend ready` line should read `workstation_name x1`. Phantom-link
warning count scales with mount count.

### 2E. Multi-env (stress)

```fish
timeout -k 5 30 env PYTHONUNBUFFERED=1 python scripts/run.py \
    robot=ar5_o6_bench_bimanual num_envs=16 headless=true max_steps=200 policy=zeros 2>&1 | tee /tmp/t2e.log
grep -E "backend ready|env ready|done after|Error|Traceback" /tmp/t2e.log
```

**Failure modes:**
- Silent hang >60s even without `| tail` → GPU/driver issue (check
  `nvidia-smi`), or another Isaac instance holding a lock, or URDF
  importer stuck on a mesh path. Retry with
  `ISAAC_KIT_LOG_VERBOSITY=verbose` for louder startup.
- URDF import errors → re-run Tier 1A, inspect phantom-link warnings.

---

## Tier 3 — Bimanual runtime rollout (PR #3 motion gate)

The only tier that flips PR #3 from "infra verified" to "verified
end-to-end." Everything before this is infrastructure that was already
working before PR #3; bimanual rollout is the new surface.

### 4A. Bimanual reach + OSC + zeros (both arms hold default pose)

```fish
python scripts/run.py controller=osc_bimanual task=bimanual_reach policy=zeros max_steps=500
```

**Critical checks:**
- `env ready: action_dim=24 observation_dim=<N>` — if action_dim ≠ 24,
  controllers/task aren't agreeing (`osc_bimanual.yaml` entries list or
  `bimanual_reach.yaml` action_dim).
- Both AR5 arms visible on bench, both Linker hands mounted.
- `[run] done after 500 steps.` with no exceptions.

### 4B. Bimanual reach + OSC + random_walk (both arms must move)

```fish
python scripts/run.py controller=osc_bimanual task=bimanual_reach policy=random_walk max_steps=1000
```

**Critical checks:**
- **Both arms visibly move.** If only one moves, action-slicing in
  `BaseEnv` is broken — inspect per-controller action slots against the
  `[arm_left(6), hand_left(6), arm_right(6), hand_right(6)]` layout.
- Both hands' fingers move (JointPD entries for `hand_left`/`hand_right`).

### 4C. Bimanual reward-trend smoke (long horizon with recorder)

```fish
python scripts/run.py controller=osc_bimanual task=bimanual_reach \
    policy=random_walk max_steps=5000 recorder=jsonl
```

Inspect reward in the JSONL output (should be finite, non-NaN,
negative, occasionally closer to zero when random-walk happens to
approach a target).

### 4D. Bimanual multi-env (B=4)

```fish
python scripts/run.py controller=osc_bimanual task=bimanual_reach \
    num_envs=4 policy=random_walk max_steps=500
```

**Failure modes unique to Tier 4:**
- `action_dim=12` instead of 24 → only one controller pair got
  instantiated. Check `len(cfg.controller.entries)` resolves to 4.
- One arm frozen while the other moves → action-slicing bug.
- Arms move but fingers don't (or vice versa) → JointPD vs OSC
  `command_dim` mismatch.
- Reward `-inf`/`nan` → ee-pose lookup failed; confirm
  `handle.frames` has both `arm_left:tool0` and `arm_right:tool0`.

---

## Tier 5 — Recorder / Replayer round-trip (GPU, ~1 min)

Verifies data-pipeline glue: `Env.step → Recorder → JSONL → Replayer →
Env.step` produces the same state trajectory.

### 5A. Record a reach episode

```fish
rm -rf outputs/test-recorder
python scripts/run.py controller=osc_bimanual task=bimanual_reach policy=random_walk \
    max_steps=400 recorder=jsonl hydra.run.dir=./outputs/test-recorder
```

Check: `outputs/test-recorder/episodes/episode_*.jsonl` exist; first
line parses as JSON with `obs`, `action`, `reward`, `terminated`,
`truncated` keys.

### 5B. Action-replay

```fish
set EPISODE (ls outputs/test-recorder/episodes/episode_*.jsonl | head -1)
python scripts/replay.py +episode=$EPISODE +mode=action_replay controller=osc_bimanual task=bimanual_reach max_steps=400
```

### 5C. State-inject

```fish
python scripts/replay.py +episode=$EPISODE +mode=state_inject controller=osc_bimanual task=bimanual_reach max_steps=400
```

---

## Tier 6 — Optional / long-horizon

### 6A. Gain tuner hot-reload

```fish
python sim/envs/test_osc/gain_tuner_osc.py --num_envs 1 --arm_role arm_left
```

Edit `sim/envs/test_osc/osc_gains.json` while running; arm behavior
should change without restart. Legacy path — see
[PR1_PROGRESS.md:373-377](PR1_PROGRESS.md#L373-L377).

### 6B. Long-horizon bimanual stability

```fish
python scripts/run.py robot=ar5_l6_bench_bimanual controller=osc_bimanual task=bimanual_reach \
    policy=random_walk max_steps=50000 num_envs=4 headless=true
```

Watch `nvidia-smi -l 2`; memory should stay flat.

---

## Blocker log

Append entries when a tier fails and isn't immediately fixable. Format:

```
- [YYYY-MM-DD] Tier X — <one-line summary>. Root cause: <…>. Fix: <…>.
```

- [2026-05-11] Tier 2A — `... | tail -40` printed nothing and process
  didn't exit. Root cause: `spawn_osc_scene.py` runs until window
  closes; in headless there's no window + no max-steps, so it runs
  forever; `tail` buffers until EOF which never comes. Fix: use the
  canonical `timeout N env PYTHONUNBUFFERED=1 ... | tee` form from
  Tier 2 above.
- [2026-05-11] Tier 2C — `timeout 20` didn't kill the process; ran
  past 30s and only emitted one `Reset robot state` line. Root cause:
  Isaac's `SimulationApp` (kit) ignores plain SIGTERM, and
  `reset_interval=600` means the second reset doesn't hit inside a
  short window. Fix: use `timeout -k 5 20` (SIGKILL 5s after SIGTERM).
  Only needing one `Reset robot state` line for pass signal is fine.
