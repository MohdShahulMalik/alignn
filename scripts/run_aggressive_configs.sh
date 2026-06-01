#!/usr/bin/env bash

set -euo pipefail

DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-123}"
BATCH_SIZE="${BATCH_SIZE:-16}"
ALIGNN_LAYERS="${ALIGNN_LAYERS:-4}"
GCN_LAYERS="${GCN_LAYERS:-4}"
LR="${LR:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.00001}"
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
    --alignn-layers "${ALIGNN_LAYERS}"
    --gcn-layers "${GCN_LAYERS}"
    --seed "${SEED}"
    --learning-rate "${LR}"
    --weight-decay "${WEIGHT_DECAY}"
    --scheduler onecycle
    --device "${DEVICE}"
    --use-cudnn-benchmark
)

run_small() {
    echo "[start] $* $(date -Is)"
    "${PYTHON_BIN}" -m alignn.cli alignn-train-small "$@"
    echo "[done] $* $(date -Is)"
}

# 1. Formation energy: plain MSE h128 with more epochs (best was 0.0887 with 60 epochs)
# Try 90 epochs to see if more training helps
run_small \
    --target formation_energy_peratom \
    "${common[@]}" \
    --hidden-dim 128 \
    --epochs 90 \
    --loss mse \
    --target-transform none \
    --run-name fe_h128_mse_e90_seed123

# 2. Formation energy: MSE with meanmax readout (captures both avg and extreme atom features)
run_small \
    --target formation_energy_peratom \
    --alignn-layers 6 \
    --gcn-layers 6 \
    "${common[@]}" \
    --hidden-dim 128 \
    --epochs 60 \
    --loss mse \
    --target-transform none \
    --readout meanmax \
    --run-name fe_h128_mse_l6_meanmax_e60_seed123

# 3. Bulk modulus: energy-mult-natoms + smoothl1 + high tail weight
# Original uses atomwise path which scales by natoms - try explicit scaling
run_small \
    --target bulk_modulus_kv \
    "${common[@]}" \
    --hidden-dim 128 \
    --epochs 30 \
    --loss smoothl1 \
    --target-transform none \
    --energy-mult-natoms \
    --high-target-threshold 200 \
    --high-positive-weight 3.0 \
    --prediction-min 0 \
    --run-name bm_h128_smoothl1_natoms_high3_e30_seed123

# 4. Bulk modulus: deeper model with MSE (match original loss)
run_small \
    --target bulk_modulus_kv \
    --alignn-layers 6 \
    --gcn-layers 6 \
    "${common[@]}" \
    --hidden-dim 128 \
    --epochs 40 \
    --loss mse \
    --target-transform none \
    --prediction-min 0 \
    --run-name bm_h128_mse_l6_e40_seed123

echo "[all done] $(date -Is)"
