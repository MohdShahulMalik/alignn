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
    --group-decay
    --scheduler onecycle
    --device "${DEVICE}"
    --use-cudnn-benchmark
)

run_small() {
    echo "[start] $* $(date -Is)"
    "${PYTHON_BIN}" -m alignn.cli alignn-train-small "$@"
    echo "[done] $* $(date -Is)"
}

# Original ALIGNN applies weight decay only to non-bias/non-norm parameters.
# These rerun the strongest bulk-modulus variants with that optimizer grouping.

run_small \
    --target bulk_modulus_kv \
    "${common[@]}" \
    --hidden-dim 128 \
    --epochs 60 \
    --loss l1 \
    --target-transform none \
    --prediction-min 0 \
    --run-name bm_h128_l1_groupdecay_e60

run_small \
    --target bulk_modulus_kv \
    "${common[@]}" \
    --hidden-dim 128 \
    --epochs 60 \
    --loss l1 \
    --target-transform none \
    --high-target-threshold 150 \
    --high-positive-weight 4.0 \
    --prediction-min 0 \
    --run-name bm_h128_l1_high4_groupdecay_e60

run_small \
    --target bulk_modulus_kv \
    "${common[@]}" \
    --hidden-dim 128 \
    --epochs 60 \
    --loss smoothl1 \
    --target-transform none \
    --high-target-threshold 200 \
    --high-positive-weight 3.0 \
    --prediction-min 0 \
    --run-name bm_h128_smoothl1_high3_groupdecay_e60

echo "[all done] $(date -Is)"
