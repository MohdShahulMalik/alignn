#!/usr/bin/env bash

set -euo pipefail

WAIT_PID="${WAIT_PID:-}"
REIMPL_START_AT="${REIMPL_START_AT:-}"
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/alignn}"

if [[ -n "${WAIT_PID}" ]]; then
    while kill -0 "${WAIT_PID}" 2>/dev/null; do
        echo "[wait] process ${WAIT_PID} still running $(date -Is)"
        sleep 300
    done
fi

export PATH="$HOME/.local/bin:$PATH"
cd "${PROJECT_ROOT}"

echo "[start] reimplementation e60 completion $(date -Is)"
START_AT="${REIMPL_START_AT}" bash scripts/run_target_specific_scratch_e60_completion.sh

echo "[start] original alignn e60 $(date -Is)"
bash scripts/run_original_alignn_e60_after_wait.sh

echo "[start] fair e60 comparison $(date -Is)"
uv run python scripts/compare_against_original_e30.py \
    --original-summary "$HOME/projects/2d-alignn/original_alignn_e60_metrics_summary.csv" \
    --output-dir results/tables/e60_compare \
    --comparison-label e60

echo "[all done] $(date -Is)"
