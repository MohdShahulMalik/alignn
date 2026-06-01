#!/usr/bin/env bash

set -euo pipefail

DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-123}"
BATCH_SIZE="${BATCH_SIZE:-16}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"

VENV_SITE_PACKAGES="${VENV_SITE_PACKAGES:-$HOME/projects/alignn/.venv/lib/python3.11/site-packages}"
export LD_LIBRARY_PATH="${VENV_SITE_PACKAGES}/nvidia/cuda_nvrtc/lib:${VENV_SITE_PACKAGES}/nvidia/cuda_runtime/lib:/opt/pytorch/cuda/lib:${LD_LIBRARY_PATH:-}"

common=(
    --dataset dft_3d
    --train-fraction 1.0
    --train-subset-size 0
    --val-subset-size 0
    --test-subset-size 0
    --batch-size "${BATCH_SIZE}"
    --alignn-layers 4
    --gcn-layers 4
    --seed "${SEED}"
    --learning-rate 0.001
    --weight-decay 0.00001
    --scheduler onecycle
    --device "${DEVICE}"
    --use-cudnn-benchmark
)

run_small() {
    echo "[start] $* $(date -Is)"
    "${PYTHON_BIN}" -m alignn.cli alignn-train-small "$@"
    echo "[done] $* $(date -Is)"
}

# === FORMATION ENERGY: SmoothL1 variants ===

# Plain SmoothL1 h64, e60
run_small \
    --target formation_energy_peratom \
    "${common[@]}" \
    --hidden-dim 64 \
    --epochs 60 \
    --loss smoothl1 \
    --target-transform none \
    --run-name fe_h64_smoothl1_e60

# Plain SmoothL1 h128, e60
run_small \
    --target formation_energy_peratom \
    "${common[@]}" \
    --hidden-dim 128 \
    --epochs 60 \
    --loss smoothl1 \
    --target-transform none \
    --run-name fe_h128_smoothl1_e60

# SmoothL1 h128 with tail weighting (earlier best recipe)
run_small \
    --target formation_energy_peratom \
    "${common[@]}" \
    --hidden-dim 128 \
    --epochs 60 \
    --loss smoothl1 \
    --target-transform none \
    --positive-weight 2 \
    --high-positive-weight 4 \
    --high-target-threshold 1.0 \
    --run-name fe_h128_smoothl1_pos2_high4_e60

# SmoothL1 h128 with lighter tail weighting
run_small \
    --target formation_energy_peratom \
    "${common[@]}" \
    --hidden-dim 128 \
    --epochs 60 \
    --loss smoothl1 \
    --target-transform none \
    --high-positive-weight 2 \
    --high-target-threshold 1.0 \
    --run-name fe_h128_smoothl1_high2_e60

echo "[all done] $(date -Is)"
