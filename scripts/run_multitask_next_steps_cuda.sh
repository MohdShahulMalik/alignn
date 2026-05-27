#!/usr/bin/env bash

set -euo pipefail

TARGETS_CSV="${TARGETS_CSV:-formation_energy_peratom,ehull,optb88vdw_bandgap,mbj_bandgap,bulk_modulus_kv}"
FINETUNE_TARGETS="${FINETUNE_TARGETS:-formation_energy_peratom mbj_bandgap bulk_modulus_kv}"
SCRATCH_TARGETS="${SCRATCH_TARGETS:-formation_energy_peratom ehull optb88vdw_bandgap mbj_bandgap bulk_modulus_kv}"
SEED="${SEED:-123}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-16}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
ALIGNN_LAYERS="${ALIGNN_LAYERS:-4}"
GCN_LAYERS="${GCN_LAYERS:-4}"
LOSS="${LOSS:-smoothl1}"
SCHEDULER="${SCHEDULER:-onecycle}"
DEVICE="${DEVICE:-cuda}"
RUN_SUFFIX="${RUN_SUFFIX:-seed${SEED}_e${EPOCHS}}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"

echo "[start] long multitask pretrain/joint $(date -Is) device=${DEVICE}"
uv run python -m alignn.cli alignn-train-multitask \
    --dataset dft_3d \
    --targets "${TARGETS_CSV}" \
    --train-fraction 1.0 \
    --train-subset-size 0 \
    --val-subset-size 0 \
    --test-subset-size 0 \
    --batch-size "${BATCH_SIZE}" \
    --hidden-dim "${HIDDEN_DIM}" \
    --alignn-layers "${ALIGNN_LAYERS}" \
    --gcn-layers "${GCN_LAYERS}" \
    --epochs "${EPOCHS}" \
    --seed "${SEED}" \
    --loss "${LOSS}" \
    --scheduler "${SCHEDULER}" \
    --run-name "longbudget_joint_all_tf1_0_${RUN_SUFFIX}" \
    --device "${DEVICE}"

pretrained="results/checkpoints/longbudget_joint_all_tf1_0_${RUN_SUFFIX}.pt"

echo "[start] long fine-tunes $(date -Is) device=${DEVICE}"
for target in ${FINETUNE_TARGETS}; do
    uv run python -m alignn.cli alignn-train-small \
        --dataset dft_3d \
        --target "${target}" \
        --train-fraction 1.0 \
        --train-subset-size 0 \
        --val-subset-size 0 \
        --test-subset-size 0 \
        --batch-size "${BATCH_SIZE}" \
        --hidden-dim "${HIDDEN_DIM}" \
        --alignn-layers "${ALIGNN_LAYERS}" \
        --gcn-layers "${GCN_LAYERS}" \
        --epochs "${EPOCHS}" \
        --seed "${SEED}" \
        --loss "${LOSS}" \
        --scheduler "${SCHEDULER}" \
        --target-transform standardize \
        --pretrained-multitask-checkpoint "${pretrained}" \
        --run-name "longbudget_finetune_${target}_tf1_0_${RUN_SUFFIX}" \
        --device "${DEVICE}"
done

echo "[start] stronger scratch baselines $(date -Is) device=${DEVICE}"
for target in ${SCRATCH_TARGETS}; do
    uv run python -m alignn.cli alignn-train-small \
        --dataset dft_3d \
        --target "${target}" \
        --train-fraction 1.0 \
        --train-subset-size 0 \
        --val-subset-size 0 \
        --test-subset-size 0 \
        --batch-size "${BATCH_SIZE}" \
        --hidden-dim "${HIDDEN_DIM}" \
        --alignn-layers "${ALIGNN_LAYERS}" \
        --gcn-layers "${GCN_LAYERS}" \
        --epochs "${EPOCHS}" \
        --seed "${SEED}" \
        --loss "${LOSS}" \
        --scheduler "${SCHEDULER}" \
        --target-transform standardize \
        --run-name "strongbaseline_scratch_${target}_tf1_0_${RUN_SUFFIX}" \
        --device "${DEVICE}"
done

echo "[start] target-group ablations $(date -Is) device=${DEVICE}"
uv run python -m alignn.cli alignn-train-multitask \
    --dataset dft_3d \
    --targets formation_energy_peratom,ehull \
    --train-fraction 1.0 \
    --train-subset-size 0 \
    --val-subset-size 0 \
    --test-subset-size 0 \
    --batch-size "${BATCH_SIZE}" \
    --hidden-dim "${HIDDEN_DIM}" \
    --alignn-layers "${ALIGNN_LAYERS}" \
    --gcn-layers "${GCN_LAYERS}" \
    --epochs "${EPOCHS}" \
    --seed "${SEED}" \
    --loss "${LOSS}" \
    --scheduler "${SCHEDULER}" \
    --run-name "group_ablation_thermodynamic_tf1_0_${RUN_SUFFIX}" \
    --device "${DEVICE}"

uv run python -m alignn.cli alignn-train-multitask \
    --dataset dft_3d \
    --targets optb88vdw_bandgap,mbj_bandgap \
    --train-fraction 1.0 \
    --train-subset-size 0 \
    --val-subset-size 0 \
    --test-subset-size 0 \
    --batch-size "${BATCH_SIZE}" \
    --hidden-dim "${HIDDEN_DIM}" \
    --alignn-layers "${ALIGNN_LAYERS}" \
    --gcn-layers "${GCN_LAYERS}" \
    --epochs "${EPOCHS}" \
    --seed "${SEED}" \
    --loss "${LOSS}" \
    --scheduler "${SCHEDULER}" \
    --run-name "group_ablation_electronic_tf1_0_${RUN_SUFFIX}" \
    --device "${DEVICE}"

echo "[analyze] $(date -Is)"
uv run python scripts/analyze_multitask_results.py
echo "[done] $(date -Is)"
