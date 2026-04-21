#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if [[ ! -x "${PROJECT_ROOT}/.venv/bin/alignn" ]]; then
    echo "Missing ${PROJECT_ROOT}/.venv/bin/alignn. Run the project environment setup first." >&2
    exit 1
fi

cuda_lib_dirs=()
for pattern in \
    "${PROJECT_ROOT}"/.venv/lib/python*/site-packages/nvidia/cuda_nvrtc/lib \
    "${PROJECT_ROOT}"/.venv/lib/python*/site-packages/nvidia/cuda_runtime/lib
do
    for dir in $pattern; do
        if [[ -d "$dir" ]]; then
            cuda_lib_dirs+=("$dir")
        fi
    done
done

if (( ${#cuda_lib_dirs[@]} > 0 )); then
    cuda_ld_path="$(IFS=:; echo "${cuda_lib_dirs[*]}")"
    export LD_LIBRARY_PATH="${cuda_ld_path}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

cd "$PROJECT_ROOT"

exec "${PROJECT_ROOT}/.venv/bin/alignn" baseline-overfit \
    --project-root "$PROJECT_ROOT" \
    --dataset dft_3d \
    --target formation_energy_peratom \
    --subset-size 16 \
    --batch-size 4 \
    --epochs 20 \
    "$@"
