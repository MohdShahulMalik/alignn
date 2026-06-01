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

# Original ALWAYS uses L1Loss (nn.L1Loss()) regardless of criterion config.
# Previous runs used MSE which was wrong.

# === FORMATION ENERGY ===
# Match original: h64, L1, e60, energy_mult_natoms=false, penalty=false
run_small \
    --target formation_energy_peratom \
    "${common[@]}" \
    --hidden-dim 64 \
    --epochs 60 \
    --loss l1 \
    --target-transform none \
    --run-name fe_match_orig_h64_l1_e60

# h128 variant with L1
run_small \
    --target formation_energy_peratom \
    "${common[@]}" \
    --hidden-dim 128 \
    --epochs 60 \
    --loss l1 \
    --target-transform none \
    --run-name fe_h128_l1_e60

# === BULK MODULUS ===
# Match original: h64, L1, e60, energy_mult_natoms=true, penalty=true
run_small \
    --target bulk_modulus_kv \
    "${common[@]}" \
    --hidden-dim 64 \
    --epochs 60 \
    --loss l1 \
    --target-transform none \
    --energy-mult-natoms \
    --penalty-factor 0.1 \
    --penalty-threshold 1.0 \
    --run-name bm_match_orig_h64_l1_e60

# h128 variant with L1
run_small \
    --target bulk_modulus_kv \
    "${common[@]}" \
    --hidden-dim 128 \
    --epochs 60 \
    --loss l1 \
    --target-transform none \
    --energy-mult-natoms \
    --penalty-factor 0.1 \
    --penalty-threshold 1.0 \
    --run-name bm_h128_l1_e60

echo "[all done] $(date -Is)"
