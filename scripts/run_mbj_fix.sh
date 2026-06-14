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

# mbj_bandgap analysis:
# - 55% are zero bandgap, model overpredicts them (+0.70 bias)
# - 7% are >5 eV, model massively underpredicts (-1.45 bias)
# - Need: downweight zeros, upweight high bandgaps, smoothl1 for robustness

# Run 1: SmoothL1 + downweight low targets + upweight high targets
run_small \
    --target mbj_bandgap \
    "${common[@]}" \
    --hidden-dim 128 \
    --epochs 60 \
    --loss smoothl1 \
    --target-transform none \
    --low-target-threshold 0.01 \
    --low-target-weight 0.3 \
    --high-target-threshold 5.0 \
    --high-positive-weight 5.0 \
    --prediction-min 0 \
    --run-name mbj_h128_smoothl1_low03_high5_e60

# Run 2: L1 + heavy high-target upweighting
run_small \
    --target mbj_bandgap \
    "${common[@]}" \
    --hidden-dim 128 \
    --epochs 60 \
    --loss l1 \
    --target-transform none \
    --low-target-threshold 0.01 \
    --low-target-weight 0.3 \
    --high-target-threshold 5.0 \
    --high-positive-weight 8.0 \
    --prediction-min 0 \
    --run-name mbj_h128_l1_low03_high8_e60

# Run 3: MSE (original uses MSE) + high-target focus
run_small \
    --target mbj_bandgap \
    "${common[@]}" \
    --hidden-dim 128 \
    --epochs 60 \
    --loss mse \
    --target-transform none \
    --low-target-threshold 0.01 \
    --low-target-weight 0.5 \
    --high-target-threshold 5.0 \
    --high-positive-weight 5.0 \
    --prediction-min 0 \
    --run-name mbj_h128_mse_low05_high5_e60

# Run 4: SmoothL1 + broader high threshold
run_small \
    --target mbj_bandgap \
    "${common[@]}" \
    --hidden-dim 128 \
    --epochs 60 \
    --loss smoothl1 \
    --target-transform none \
    --low-target-threshold 0.01 \
    --low-target-weight 0.3 \
    --high-target-threshold 2.0 \
    --high-positive-weight 4.0 \
    --prediction-min 0 \
    --run-name mbj_h128_smoothl1_low03_high4_e60

echo "[all done] $(date -Is)"
