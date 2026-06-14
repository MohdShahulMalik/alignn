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

# Importance sampling bulk modulus: oversample negatives and very-high values
# Negatives (6 samples) and very-high >=300 (9 samples) are extremely rare

# Moderate oversampling: neg 5x, vhigh 3x
run_small \
    --target bulk_modulus_kv \
    "${common[@]}" \
    --hidden-dim 128 \
    --epochs 60 \
    --loss l1 \
    --target-transform none \
    --prediction-min 0 \
    --importance-sample \
    --importance-sample-bins "neg:5.0,vhigh:3.0" \
    --run-name bm_is_l1_neg5_vhigh3_e60

# Heavy oversampling: neg 8x, vhigh 5x
run_small \
    --target bulk_modulus_kv \
    "${common[@]}" \
    --hidden-dim 128 \
    --epochs 60 \
    --loss l1 \
    --target-transform none \
    --prediction-min 0 \
    --importance-sample \
    --importance-sample-bins "neg:8.0,vhigh:5.0" \
    --run-name bm_is_l1_neg8_vhigh5_e60

# High region oversampling too: neg 5x, high 2x, vhigh 4x
run_small \
    --target bulk_modulus_kv \
    "${common[@]}" \
    --hidden-dim 128 \
    --epochs 60 \
    --loss l1 \
    --target-transform none \
    --prediction-min 0 \
    --importance-sample \
    --importance-sample-bins "neg:5.0,high:2.0,vhigh:4.0" \
    --run-name bm_is_l1_neg5_high2_vhigh4_e60

# Combined: importance sampling + smoothl1 + high-target upweighting
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
    --importance-sample \
    --importance-sample-bins "neg:5.0,vhigh:4.0" \
    --run-name bm_is_smoothl1_high3_neg5_vhigh4_e60

echo "[all done] $(date -Is)"
