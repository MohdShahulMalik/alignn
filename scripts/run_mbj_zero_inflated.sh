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

# Zero-inflated mbj_bandgap: classifier + regression, L1 regression loss
# 552/1000 samples are zero bandgap — classifier learns P(bandgap > 0)
run_small \
    --target mbj_bandgap \
    "${common[@]}" \
    --hidden-dim 128 \
    --epochs 60 \
    --loss l1 \
    --target-transform none \
    --zero-inflated \
    --prediction-min 0 \
    --run-name mbj_zi_l1_e60

# Zero-inflated + SmoothL1 regression (robust to outliers in nonzero tail)
run_small \
    --target mbj_bandgap \
    "${common[@]}" \
    --hidden-dim 128 \
    --epochs 60 \
    --loss smoothl1 \
    --target-transform none \
    --zero-inflated \
    --prediction-min 0 \
    --run-name mbj_zi_smoothl1_e60

# Zero-inflated + higher classifier weight (emphasize correct zero/nonzero detection)
run_small \
    --target mbj_bandgap \
    "${common[@]}" \
    --hidden-dim 128 \
    --epochs 60 \
    --loss l1 \
    --target-transform none \
    --zero-inflated \
    --zero-inflated-classifier-weight 2.0 \
    --prediction-min 0 \
    --run-name mbj_zi_l1_clsw2_e60

# Zero-inflated + lower regression weight (let classifier dominate early)
run_small \
    --target mbj_bandgap \
    "${common[@]}" \
    --hidden-dim 128 \
    --epochs 60 \
    --loss l1 \
    --target-transform none \
    --zero-inflated \
    --zero-inflated-regression-weight 0.5 \
    --prediction-min 0 \
    --run-name mbj_zi_l1_regw05_e60

echo "[all done] $(date -Is)"
