#!/usr/bin/env bash

set -euo pipefail

DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-123}"
BATCH_SIZE="${BATCH_SIZE:-64}"
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

# h256 + MSE + 60 epochs (matching original ALIGNN hyperparameters)
run_small \
    --target bulk_modulus_kv \
    "${common[@]}" \
    --hidden-dim 256 \
    --epochs 60 \
    --loss mse \
    --target-transform none \
    --prediction-min 0 \
    --keep-data-order \
    --run-name bm_h256_mse_e60

# h256 + MSE + 60 epochs + group_decay (original optimizer grouping)
run_small \
    --target bulk_modulus_kv \
    "${common[@]}" \
    --hidden-dim 256 \
    --epochs 60 \
    --loss mse \
    --target-transform none \
    --prediction-min 0 \
    --keep-data-order \
    --group-decay \
    --run-name bm_h256_mse_groupdecay_e60

# h256 + L1 + 60 epochs (compare MSE vs L1 at h256)
run_small \
    --target bulk_modulus_kv \
    "${common[@]}" \
    --hidden-dim 256 \
    --epochs 60 \
    --loss l1 \
    --target-transform none \
    --prediction-min 0 \
    --keep-data-order \
    --run-name bm_h256_l1_e60

# h256 + SmoothL1 + 60 epochs
run_small \
    --target bulk_modulus_kv \
    "${common[@]}" \
    --hidden-dim 256 \
    --epochs 60 \
    --loss smoothl1 \
    --target-transform none \
    --prediction-min 0 \
    --keep-data-order \
    --run-name bm_h256_smoothl1_e60

echo "[all done] $(date -Is)"
