#!/usr/bin/env bash

set -euo pipefail

DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-123}"
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

# =====================================================================
# Bulk modulus: h128 + dropout + early stopping + gradient clipping
# Root cause: h256 overfits due to zero dropout; h128 peaks at epoch ~12
# =====================================================================

# Config 1: h128, dropout=0.1, early_stop=15, grad_clip=1.0
run_small \
    --target bulk_modulus_kv \
    "${common[@]}" \
    --hidden-dim 128 \
    --batch-size 64 \
    --epochs 60 \
    --loss l1 \
    --target-transform none \
    --prediction-min 0 \
    --keep-data-order \
    --dropout 0.1 \
    --early-stopping-patience 15 \
    --max-grad-norm 1.0 \
    --run-name bm_h128_drop01_es15_clip1_e60

# Config 2: h128, dropout=0.2, early_stop=15, grad_clip=1.0
run_small \
    --target bulk_modulus_kv \
    "${common[@]}" \
    --hidden-dim 128 \
    --batch-size 64 \
    --epochs 60 \
    --loss l1 \
    --target-transform none \
    --prediction-min 0 \
    --keep-data-order \
    --dropout 0.2 \
    --early-stopping-patience 15 \
    --max-grad-norm 1.0 \
    --run-name bm_h128_drop02_es15_clip1_e60

# Config 3: h128, dropout=0.1, early_stop=10, grad_clip=1.0, weight_decay=0.001
run_small \
    --target bulk_modulus_kv \
    "${common[@]}" \
    --hidden-dim 128 \
    --batch-size 64 \
    --epochs 60 \
    --loss l1 \
    --target-transform none \
    --prediction-min 0 \
    --keep-data-order \
    --dropout 0.1 \
    --early-stopping-patience 10 \
    --max-grad-norm 1.0 \
    --weight-decay 0.001 \
    --run-name bm_h128_drop01_es10_wd001_e60

# Config 4: h128, dropout=0.15, early_stop=15, grad_clip=0.5
run_small \
    --target bulk_modulus_kv \
    "${common[@]}" \
    --hidden-dim 128 \
    --batch-size 64 \
    --epochs 60 \
    --loss l1 \
    --target-transform none \
    --prediction-min 0 \
    --keep-data-order \
    --dropout 0.15 \
    --early-stopping-patience 15 \
    --max-grad-norm 0.5 \
    --run-name bm_h128_drop015_es15_clip05_e60

# =====================================================================
# mbj_bandgap: h128 + zero-inflated + dropout + early stopping
# Root cause: massive overfitting (train MAE 0.07 vs val MAE 1.32)
# =====================================================================

# Config 5: zi l1, dropout=0.1, early_stop=15
run_small \
    --target mbj_bandgap \
    "${common[@]}" \
    --hidden-dim 128 \
    --batch-size 16 \
    --epochs 60 \
    --loss l1 \
    --target-transform none \
    --zero-inflated \
    --prediction-min 0 \
    --dropout 0.1 \
    --early-stopping-patience 15 \
    --max-grad-norm 1.0 \
    --run-name mbj_zi_l1_drop01_es15_e60

# Config 6: zi l1, dropout=0.2, early_stop=15
run_small \
    --target mbj_bandgap \
    "${common[@]}" \
    --hidden-dim 128 \
    --batch-size 16 \
    --epochs 60 \
    --loss l1 \
    --target-transform none \
    --zero-inflated \
    --prediction-min 0 \
    --dropout 0.2 \
    --early-stopping-patience 15 \
    --max-grad-norm 1.0 \
    --run-name mbj_zi_l1_drop02_es15_e60

# Config 7: zi l1, dropout=0.1, early_stop=10, weight_decay=0.001
run_small \
    --target mbj_bandgap \
    "${common[@]}" \
    --hidden-dim 128 \
    --batch-size 16 \
    --epochs 60 \
    --loss l1 \
    --target-transform none \
    --zero-inflated \
    --prediction-min 0 \
    --dropout 0.1 \
    --early-stopping-patience 10 \
    --max-grad-norm 1.0 \
    --weight-decay 0.001 \
    --run-name mbj_zi_l1_drop01_es10_wd001_e60

# Config 8: zi l1, dropout=0.15, early_stop=15, grad_clip=0.5
run_small \
    --target mbj_bandgap \
    "${common[@]}" \
    --hidden-dim 128 \
    --batch-size 16 \
    --epochs 60 \
    --loss l1 \
    --target-transform none \
    --zero-inflated \
    --prediction-min 0 \
    --dropout 0.15 \
    --early-stopping-patience 15 \
    --max-grad-norm 0.5 \
    --run-name mbj_zi_l1_drop015_es15_clip05_e60

echo "[all done] $(date -Is)"
