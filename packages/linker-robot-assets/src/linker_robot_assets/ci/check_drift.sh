#!/usr/bin/env bash
# CI drift check: re-run the composer on every workstation and verify that
# the committed workstation.urdf / workstation.mjcf / manifest.yaml match a
# fresh compose. Any mismatch means someone edited a recipe or component
# source without re-running the composer.
#
# XRDF coverage: the composer's component-level hash already includes any
# variant XRDF (see compose.py `_hash_component`). An XRDF edit changes
# the manifest's component hash → manifest text changes → drift caught
# transitively. So this script does not need a separate XRDF arm.
#
# Usage (run from anywhere — paths resolve relative to this script):
#   bash packages/linker-robot-assets/src/linker_robot_assets/ci/check_drift.sh              # check all workstations
#   bash packages/linker-robot-assets/src/linker_robot_assets/ci/check_drift.sh <ws_name>    # check one workstation
#
# Optionally also enforce semver on joint-name renames vs the previous
# git tag (refuses minor/patch bumps when joint names rename):
#   CHECK_JOINT_RENAMES=1 bash …/check_drift.sh
#
# Exit code: 0 if clean, 1 if drift / version violation, 2 on usage errors.
#
# Requires: linker-robot-assets[authoring] installed in the active env, OR
# packages/linker-robot-assets/src on PYTHONPATH (a fresh checkout sets
# this via pyproject.toml's [tool.pytest.ini_options].pythonpath; for
# direct invocation outside pytest, prefix with PYTHONPATH=… ).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Layout: HERE = packages/linker-robot-assets/src/linker_robot_assets/ci/
#         REPO_ROOT = HERE/../../../../.. (climb 5 levels)
REPO_ROOT="$(cd "${HERE}/../../../../.." && pwd)"
WORKSTATIONS_DIR="${HERE}/../assets/workstations"
PYTHON="${PYTHON:-python3}"

if [[ $# -gt 1 ]]; then
    echo "usage: $0 [workstation_name]" >&2
    exit 2
fi

if [[ $# -eq 1 ]]; then
    TARGETS=("${WORKSTATIONS_DIR}/$1")
else
    shopt -s nullglob
    TARGETS=()
    for d in "${WORKSTATIONS_DIR}"/*/; do
        if [[ -f "${d}/recipe.yaml" ]]; then
            TARGETS+=("${d%/}")
        fi
    done
fi

if [[ ${#TARGETS[@]} -eq 0 ]]; then
    echo "no workstations found under ${WORKSTATIONS_DIR}" >&2
    exit 2
fi

cd "${REPO_ROOT}"

fail=0
for ws in "${TARGETS[@]}"; do
    name="$(basename "${ws}")"
    echo "[check_drift] ${name}"
    if ! "${PYTHON}" -m linker_robot_assets.composer.compose "${ws}" --check-drift; then
        fail=1
    fi
done

if [[ ${fail} -ne 0 ]]; then
    echo >&2
    echo "drift detected. Re-run:" >&2
    echo "  for ws in packages/linker-robot-assets/src/linker_robot_assets/assets/workstations/*/; do python -m linker_robot_assets.composer.compose \"\$ws\"; done" >&2
    echo "and commit the updated artifacts." >&2
    exit 1
fi
echo "all workstations clean."

# Optional semver / joint-rename guard against the previous git tag.
# Off by default so locally invoking the drift check stays cheap.
if [[ "${CHECK_JOINT_RENAMES:-0}" == "1" ]]; then
    echo "[check_drift] checking joint-name stability vs previous tag…"
    if ! "${PYTHON}" -m linker_robot_assets.ci.joint_rename_guard; then
        exit 1
    fi
fi
