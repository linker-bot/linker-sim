#!/usr/bin/env bash
# CI drift check: re-run the composer on every workstation and verify that
# the committed workstation.urdf / workstation.mjcf / manifest.yaml match a
# fresh compose. Any mismatch means someone edited a recipe or component
# source without re-running the composer.
#
# Usage:
#   bash tools/ci/check_drift.sh              # check all workstations
#   bash tools/ci/check_drift.sh <ws_name>    # check one workstation
#
# Exit code: 0 if clean, 1 if drift, 2 on usage errors.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"
WORKSTATIONS_DIR="${REPO_ROOT}/assets/workstations"
PYTHON="${PYTHON:-python3}"

cd "${REPO_ROOT}"

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

fail=0
for ws in "${TARGETS[@]}"; do
    name="$(basename "${ws}")"
    echo "[check_drift] ${name}"
    if ! "${PYTHON}" -m tools.composer.compose "${ws}" --check-drift; then
        fail=1
    fi
done

if [[ ${fail} -ne 0 ]]; then
    echo >&2
    echo "drift detected. Re-run:" >&2
    echo "  for ws in assets/workstations/*/; do python -m tools.composer.compose \"\$ws\"; done" >&2
    echo "and commit the updated artifacts." >&2
    exit 1
fi
echo "all workstations clean."
