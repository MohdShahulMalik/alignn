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

# === TARGET 1: formation_energy_peratom ===
# Best: MAE=0.0838 (beats original 0.0853)
# Config: h128, L1, 60 epochs
run_small \
    --target formation_energy_peratom \
    "${common[@]}" \
    --hidden-dim 128 \
    --epochs 60 \
    --loss l1 \
    --target-transform none \
    --run-name best5_fe_h128_l1_e60

# === TARGET 2: bulk_modulus_kv ===
# Best: MAE=14.20 (loses to original 13.58)
# Config: h128, L1, 60 epochs, no energy_mult_natoms, prediction-min 0
run_small \
    --target bulk_modulus_kv \
    "${common[@]}" \
    --hidden-dim 128 \
    --epochs 60 \
    --loss l1 \
    --target-transform none \
    --prediction-min 0 \
    --run-name best5_bm_h128_l1_e60

# === TARGET 3: ehull ===
# Best: MAE=0.0273 (beats original 0.0586)
# Config: h128, smoothl1, 30 epochs, log1p, tail weighting
run_small \
    --target ehull \
    "${common[@]}" \
    --hidden-dim 128 \
    --epochs 30 \
    --loss smoothl1 \
    --target-transform log1p \
    --positive-weight 0.35 \
    --low-target-threshold 0.001 \
    --low-target-weight 6.0 \
    --high-target-threshold 0.1 \
    --high-positive-weight 0.2 \
    --prediction-min 0 \
    --run-name best5_ehull_h128_log1p_e30

# === TARGET 4: optb88vdw_bandgap ===
# Best: MAE=0.2281 (beats original 0.2426)
# Config: h128, smoothl1, 35 epochs, tail weighting
run_small \
    --target optb88vdw_bandgap \
    "${common[@]}" \
    --hidden-dim 128 \
    --epochs 35 \
    --loss smoothl1 \
    --target-transform none \
    --low-target-threshold 0.05 \
    --low-target-weight 2.0 \
    --high-target-threshold 1.0 \
    --high-positive-weight 4.0 \
    --prediction-min 0 \
    --run-name best5_optbg_h128_e35

# === TARGET 5: mbj_bandgap ===
# Best scratch: MAE=0.8231 (loses to original 0.8178)
# Config: h128, mse, 50 epochs, prediction-min 0
run_small \
    --target mbj_bandgap \
    "${common[@]}" \
    --hidden-dim 128 \
    --epochs 50 \
    --loss mse \
    --target-transform none \
    --prediction-min 0 \
    --run-name best5_mbj_h128_mse_e50

echo "[all done] $(date -Is)"
