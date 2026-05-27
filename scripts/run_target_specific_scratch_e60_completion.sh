#!/usr/bin/env bash

set -euo pipefail

DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-123}"
BATCH_SIZE="${BATCH_SIZE:-16}"
ALIGNN_LAYERS="${ALIGNN_LAYERS:-4}"
GCN_LAYERS="${GCN_LAYERS:-4}"
LR="${LR:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.00001}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"

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
)

run_small() {
    echo "[start] $* $(date -Is)"
    uv run python -m alignn.cli alignn-train-small "$@"
    echo "[done] $* $(date -Is)"
}

# These complete the fair longer-budget set. The first sweep already contains
# formation-energy and bulk-modulus 60-epoch variants; this script adds 60-epoch
# variants for the targets that were only 30/35/50 epochs there.

run_small \
    --target ehull \
    "${common[@]}" \
    --hidden-dim 128 \
    --epochs 60 \
    --loss smoothl1 \
    --target-transform log1p \
    --positive-weight 0.35 \
    --low-target-threshold 0.001 \
    --low-target-weight 6.0 \
    --high-target-threshold 0.1 \
    --high-positive-weight 0.2 \
    --prediction-min 0 \
    --run-name improve_scratch_ehull_h128_log1p_low6_downhigh_e60_seed123

run_small \
    --target optb88vdw_bandgap \
    "${common[@]}" \
    --hidden-dim 128 \
    --epochs 60 \
    --loss smoothl1 \
    --target-transform none \
    --low-target-threshold 0.05 \
    --low-target-weight 2.0 \
    --high-target-threshold 1.0 \
    --high-positive-weight 4.0 \
    --prediction-min 0 \
    --run-name improve_scratch_optb88vdw_bandgap_h128_low2_high4_clamp_e60_seed123

run_small \
    --target mbj_bandgap \
    "${common[@]}" \
    --hidden-dim 128 \
    --epochs 60 \
    --loss smoothl1 \
    --target-transform none \
    --low-target-threshold 0.05 \
    --low-target-weight 2.0 \
    --high-target-threshold 1.0 \
    --high-positive-weight 4.0 \
    --prediction-min 0 \
    --run-name improve_scratch_mbj_bandgap_h128_low2_high4_clamp_e60_seed123

run_small \
    --target mbj_bandgap \
    "${common[@]}" \
    --hidden-dim 128 \
    --epochs 60 \
    --loss mse \
    --target-transform none \
    --prediction-min 0 \
    --run-name improve_scratch_mbj_bandgap_h128_mse_clamp_e60_seed123

uv run python scripts/analyze_multitask_results.py
echo "[all done] $(date -Is)"
