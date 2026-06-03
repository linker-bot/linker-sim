# Refactor + Open-Source Plan

**Status**: locked, ready to execute
**Last updated**: 2026-06-03
**Owner**: Ryan Zhou

This document is the cross-session reference for the multi-phase refactor that
takes `dex-tool-rl` from "internal mixed-purpose repo" to a public, packaged,
multi-consumer codebase. Read top-to-bottom before resuming work.

---

## 1. Context

### 1.1 Stakeholders

| # | Stakeholder | Consumes |
|---|---|---|
| 1 | UMI-Dex (already public, `~/codes/UMI-Dex`) | Independent; this repo replays its bag data |
| 2 | This repo (`linker-sim`, soon-to-be public) | Sim runtime + RL + replay |
| 3 | Real-robot data collection team (internal) | Sim replay backends, leaning toward Viser |
| 4 | Algorithms team (internal) | Pre-configured controllers + sim env for RL training (aspirational, not training yet) |

### 1.2 Coupling problems being solved

- `scripts/umi_bag_to_ee_poses.py:59-61` and `scripts/anchor_search.py:13` hardcode `~/codes/UMI-Dex` into `sys.path`. Breaks for anyone without that exact local checkout.
- `assets/` (URDF/MJCF/XRDF + composer) is the most-shared product but is not in the wheel (`pyproject.toml` packages only `["sim", "tools"]`).
- URDF/MJCF/XRDF triplication: composer generates URDF→MJCF deterministically with CI drift checks; XRDF is hand-authored only for `ar5_08`. Drift risk.
- Internal docs and customer-named hardware references currently live in `docs/`.
- No `LICENSE`, no `THIRD_PARTY_NOTICES.md` — blocks public release.

---

## 2. Locked-in decisions

1. **Repo rename**: `dex-tool-rl` → `linker-sim`.
2. **New package**: `linker-robot-assets` — bundles asset data + composer + validators + hand decoders + loader API. Composer is lazy-imported behind `[authoring]` extra. (Option A from the design discussion: composer ships with the assets, not as a separate repo.)
3. **Workspace layout** (Option a, symmetric):
   ```
   linker-sim/                      (repo root)
   ├── pyproject.toml               (uv workspace root, no source)
   ├── packages/
   │   ├── linker-sim/              (runtime: was sim/, tools/)
   │   │   ├── pyproject.toml
   │   │   └── src/linker_sim/
   │   └── linker-robot-assets/     (assets + composer + decoders)
   │       ├── pyproject.toml
   │       ├── src/linker_robot_assets/
   │       └── assets/
   └── apps/                        (Hydra entry scripts: run.py, replay.py, etc.)
   ```
4. **Viser as a third backend** behind `[viser]` extra. Replay-only first, teleop deferred.
5. **Hand decoder** in `linker_robot_assets/decoders/hand.py` — linear interpolation from SDK [0, 100] → URDF `[lower, upper]`. Marked `CONVENTION = "linear-fit-v0"`. Documented as unstable; revisit when SDK lands an angle convention.
6. **License**: Apache 2.0 (matches UMI-Dex). `THIRD_PARTY_NOTICES.md` follows the Isaac Lab NOTICES shape.
7. **UMI-Dex PyPI publish deferred**. UMI-Dex is WIP. Path hack stays for now, flagged with TODO comments and a `docs/known_limitations.md` entry.
8. **Public/private overlay deferred**. All current assets are open-source-clear. Add overlay mechanism only when there is unreleased hardware to keep out of the public log.
9. **Algo-team APIs not frozen**. Controllers / envs / tasks can be refactored freely until they have working training runs.
10. **Personal email addresses kept in git history**. No history rewrite. Three identities total — see §6.
11. **Mesh LFS**: existing `*.STL`/`*.stl` LFS coverage retained. Widen `.gitattributes` to other binary mesh formats proactively (`*.obj`, `*.dae`, `*.glb`, `*.usd*`, `*.fbx`).

---

## 3. Licensing summary

Source: NVIDIA's own license tiers (Isaac Lab uses BSD-3-Clause; Isaac Sim is dual-licensed; cuRobo is proprietary). See `license_research.md` at repo root.

| Dep | License | Treatment in `linker-sim` |
|---|---|---|
| Isaac Lab | BSD-3-Clause | Optional `[isaac]` extra. Cite Mittal et al. (Orbit paper) in NOTICES. |
| Isaac Sim (Apache wrapper) | Apache 2.0 | Optional `[isaac]` extra. |
| Isaac Sim (Omniverse Kit binaries) | NVIDIA Additional Software and Materials License | Not redistributed; users install themselves. README discloses EULA. |
| cuRobo | Proprietary EULA | Confirmed not used in runtime. Drop from NOTICES. |
| MuJoCo | Apache 2.0 | Optional `[mujoco]` extra. Standard attribution. |
| Viser | Apache 2.0 | Optional `[viser]` extra. Standard attribution. |
| UMI-Dex | Apache 2.0 | Optional `[umi-replay]` extra (deferred to PyPI). |
| Rokae meshes | Open-source per project owner | Attribution in NOTICES. |
| Linkerhand meshes | Open-source per project owner | Attribution in NOTICES. |

**Rule**: do not bundle any NVIDIA binary, USD asset originating from Isaac Sim sample content, or cuRobo files. Phase 0.4 verifies.

---

## 4. The Plan

Five phases. Each step has a "done when" check. Phase 0 then Phase 1 is the recommended order; Phases 2–4 can interleave once the workspace exists.

### Phase 0 — Pre-publish gates *(must complete before first public push)*

| # | Step | Done when |
|---|---|---|
| 0.1 | Move `docs/PR{1,2,3}_PROGRESS.md`, `target_spec.md`, `UMI_REPLAY_PLAN.md` to a private wiki or `_internal/` (gitignored). Distill load-bearing context into a clean `docs/architecture.md`. | `git ls-files docs/` shows no `PR*_PROGRESS.md` or `target_spec.md`. |
| 0.2 | Fix hardcoded `/home/zhy/...` in `docs/TEST_PIPELINE.md:12` and `docs/PR2_PROGRESS.md:261`. | `grep -r "/home/zhy" docs/ scripts/` returns nothing. |
| 0.3 | Add `LICENSE` (Apache 2.0). | File exists, matches UMI-Dex's `LICENSE` shape. |
| 0.4 | Audit cuRobo + USD usage. Grep for `import curobo`, `from curobo`, `*.usd`, `*.usda`, `*.usdc`. Decide per finding: keep behind `[curobo]` extra (none expected), or remove. | Audit doc lists every hit and disposition. Confirms no NVIDIA-sourced USDs in `assets/`. |
| 0.5 | Write `THIRD_PARTY_NOTICES.md`. Entries per §3. Skip cuRobo if 0.4 confirms zero usage. | File renders cleanly; every runtime dep attributed. |
| 0.6 | Add `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `CODEOWNERS`, `.github/ISSUE_TEMPLATE/`, `.github/PULL_REQUEST_TEMPLATE.md`. | UMI-Dex equivalents copied and adapted. |
| 0.7 | Sweep for secrets: `git log -p \| grep -iE 'token\|secret\|password\|api_key\|bearer'`. No history rewrite — see §6. | Grep returns no live secrets. |
| 0.8 | Delete untracked-but-staged-prone root files: `data.json`, `ik_replay_tracking.png`, `MUJOCO_LOG.TXT`, any `episode_json/` or `outputs/` lingering at root. | `git status` clean except for intentional changes. |
| 0.9 | Commit a tiny synthetic fixture for `source=data_collection` happy path (the `episode_000025` default points at gitignored data). Promote the synthesis pattern from `tests/test_replay_a7_lite.py`. | Fresh clone runs `scripts/replay.py source=data_collection` end-to-end without external data. |
| 0.10 | Widen `.gitattributes` LFS coverage: add `*.obj`, `*.dae`, `*.glb`, `*.usd*`, `*.usdc`, `*.usda`, `*.fbx`. | `cat .gitattributes` shows the new lines; `git check-attr filter <foo>.obj` reports `lfs`. |

### Phase 1 — Repo rename + workspace skeleton

| # | Step | Done when |
|---|---|---|
| 1.1 | Rename GitHub repo `dex-tool-rl` → `linker-sim`. Update local remotes. | `git remote -v` shows `linker-sim`. Old URL still resolves via GitHub redirect. |
| 1.2 | Update `pyproject.toml`: `[project] name = "linker-sim"`. Search and replace `dex-tool-rl` → `linker-sim` in docs, configs, READMEs. Search for any Python module named `dex_tool_rl` (likely none). | `grep -r dex-tool-rl` returns nothing except CHANGELOG entries describing the rename. |
| 1.3 | Convert root `pyproject.toml` to a `uv` workspace root with two members: `packages/linker-sim/` and `packages/linker-robot-assets/`. Move `sim/` and `tools/` into `packages/linker-sim/src/linker_sim/`. Apps stay at repo root in `apps/`. | `uv sync` resolves both members. `pytest` still passes. |
| 1.4 | Update local references: `~/codes/dex-tool-rl` → `~/codes/linker-sim` in any local config or memory entries. Update CLAUDE.md memory `umi_dex_integration.md` to reflect new layout. | Memory entries match reality. |

### Phase 2 — Extract `linker-robot-assets`

| # | Step | Done when |
|---|---|---|
| 2.1 | `git mv assets/` → `packages/linker-robot-assets/assets/`. `git mv tools/composer/`, `tools/validate_workstation.py`, `tools/ci/check_drift.sh` → `packages/linker-robot-assets/src/linker_robot_assets/composer/`. | Files relocated; git history preserved. |
| 2.2 | Write loader API: `linker_robot_assets/__init__.py` with `asset_root() -> Path`, `load_manifest(name) -> dict`, `workstations() -> list[str]`. Light imports only (pyyaml). Composer imports gated behind `from linker_robot_assets.composer import ...`. | `python -c "import linker_robot_assets; linker_robot_assets.asset_root()"` works without pulling jsonschema. |
| 2.3 | Configure wheel to force-include `assets/` via hatch `force-include`. CI must `git lfs pull` before `python -m build`. | `unzip -l dist/*.whl \| grep workstation.urdf` lists all 11 workstations and meshes are real bytes, not LFS pointers. |
| 2.4 | Add `[authoring]` extra: `jsonschema`, etc. Default install minimal. | `pip install linker-robot-assets` works without authoring extras; `pip install linker-robot-assets[authoring]` enables `python -m linker_robot_assets.composer.compose`. |
| 2.5 | Update `linker_sim/registry.py` to call `linker_robot_assets.asset_root()` instead of hardcoding `assets/`. | `pytest packages/linker-sim/` passes; `apps/run.py` resolves assets through the package. |
| 2.6 | Update `add_robot_component` skill instructions for the new path. | Adding a dummy component end-to-end via the skill works against the new layout. |
| 2.7 | CI gate: extend `check_drift.sh` to cover XRDF (where present) and to compare manifest joint sets against the previous tag. Refuse minor/patch version bumps when joint names rename. | CI fails if you rename a joint and bump only patch. |

### Phase 3 — Viser backend

| # | Step | Done when |
|---|---|---|
| 3.1 | Add `[viser]` extra: `viser`, `numpy`, `trimesh`. | `pip install linker-sim[viser]` succeeds in a clean env. |
| 3.2 | Create `packages/linker-sim/src/linker_sim/backends/viser/` implementing the same `WorkstationHandle` contract as `mujoco`/`isaac` backends. Replay-only first; teleop later if needed. | A bag replay command (`scripts/replay.py backend=viser source=...`) opens a Viser scene that animates the robot. |
| 3.3 | Write `tests/test_viser_replay.py` mirroring the existing replay test pattern. | Test passes in CI. |
| 3.4 | Document the data-collection team's install path: `pip install linker-sim[viser,umi-replay]` (later — for now, `[viser]` only). | README has a "Data collection team quick start" section. |

### Phase 4 — Hand decoder

| # | Step | Done when |
|---|---|---|
| 4.1 | Create `linker_robot_assets/decoders/hand.py` with `decode_hand(name, side, sdk_0_100) -> joint_angles` doing linear interp from URDF `[lower, upper]`. Export `CONVENTION = "linear-fit-v0"`. Module docstring carries the SDK-pending warning. | `decode_hand("linkerhand_l6", "right", np.zeros(6))` returns the lower-limit vector. |
| 4.2 | Add `decoder.yaml` next to each hand component: channel→joint name mapping, optional per-joint clip overrides. | Three hands have working `decoder.yaml`. |
| 4.3 | Switch the UMI replay path in `apps/umi_bag_to_ee_poses.py` (formerly `scripts/`) to call `decode_hand` for hand telemetry channels. | Re-running existing bag → ee_poses on a known bag produces hand joints in the URDF range. |
| 4.4 | Add `docs/known_limitations.md` with sections "Hand decoder linear fit" and "UMI-Dex path hack" (links to deferred items). | File exists, README links to it. |
| 4.5 | Stamp `decoder_convention: linear-fit-v0` into the replay output so downstream knows. | Output telemetry npz has the field. |

### Phase 5 — Defer list (flagged, not blocking)

| # | Item | Trigger to act |
|---|---|---|
| 5.1 | Publish `umi-dex` to PyPI / internal index; kill path hack in `apps/umi_bag_to_ee_poses.py` and `apps/anchor_search.py`. Replace with `umi-replay = ["umi-dex>=X.Y"]` extra. | UMI-Dex's `umi_dex` Python API stabilizes (Ryan decides). |
| 5.2 | Replace hand-decoder linear fit with SDK-defined convention. Bump `CONVENTION = "sdk-vN"`. Grep all bagged data for the old constant. | Linkerbot SDK ships an angle convention. |
| 5.3 | Public/private overlay mechanism (composer reads `LINKER_SIM_OVERLAY=/path` for additional asset roots). | First time there is unreleased hardware to keep out of the public log. |
| 5.4 | XRDF as third generated artifact (rather than hand-authored). | When more than the current 1 robot needs XRDF; `ar5_08` is the only one with XRDFs today. |
| 5.5 | Freeze controller / env / task public APIs (`api_version`, deprecation policy). | Algo team has a working RL training run dependent on those APIs. |

For 5.1, drop a comment block at the top of the path-hack scripts so they self-document the TODO:

```python
# TODO(linker-sim): replace with `from umi_dex...` once umi-dex is published
# to PyPI / internal index. Tracking: docs/known_limitations.md#umi-dex-path-hack
```

---

## 5. Open items resolved

- **cuRobo in runtime?** No (per project owner). Phase 0.4 verifies and drops the entry from NOTICES if confirmed.
- **Mesh LFS**: already configured for `*.STL`/`*.stl`, 161 files. Widen in Phase 0.10.
- **Email policy**: personal addresses retained, no history rewrite. See §6.
- **Workspace layout**: Option (a), both packages under `packages/`.

---

## 6. Git history committer identities

```
Ryan Zhou      <ryanzed12138@gmail.com>
zhaomeihan     <2750058081@qq.com>
周浩宇         <14+zhouhaoyu@noreply.localhost>
```

The third (`noreply.localhost`) looks like a local-git-config artifact. Confirm with the contributor whether it was intentional (privacy mask = harmless) or a misconfiguration before publish.

---

## 7. Reference points

- Source repos: `~/codes/dex-tool-rl` (this), `~/codes/UMI-Dex` (UMI-Dex)
- Existing docs that informed the plan: `REPO_OVERVIEW.md`, `docs/UMI_REPLAY_PLAN.md`, `docs/TEST_PIPELINE.md`, `docs/urdf_assets_infra.md`, `pyproject.toml`
- License research: `license_research.md` at repo root
- Path-hack lines to kill: `scripts/umi_bag_to_ee_poses.py:59-61`, `scripts/anchor_search.py:13`
- Asset corpus today: `assets/` (206 MB, 161 mesh files in LFS, 11 workstations, 7 components)
- Backend pattern to mirror for Viser: `sim/backends/{mujoco,isaac}/`

---

## 8. How to resume

If picking this up in a new session:

1. Read this file top-to-bottom.
2. Check what's been done: `git log --oneline -20`, look for commits referencing Phase numbers.
3. Confirm the locked-in decisions in §2 still hold; if anything changed, update this file first before coding.
4. Pick up at the next unchecked step in §4.

When checking off a step, update the "done when" column to reference the commit SHA or PR that completed it. Keep this file truthful.
