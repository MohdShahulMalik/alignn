#!/usr/bin/env bash

set -euo pipefail

TARGETS="${TARGETS:-formation_energy_peratom ehull optb88vdw_bandgap mbj_bandgap bulk_modulus_kv}"
FRACTIONS="${FRACTIONS:-0.10 0.25 0.50 1.0}"
SEED="${SEED:-123}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:?Set PRETRAINED_CHECKPOINT to a multi-task .pt checkpoint}"
BATCH_SIZE="${BATCH_SIZE:-16}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
ALIGNN_LAYERS="${ALIGNN_LAYERS:-4}"
GCN_LAYERS="${GCN_LAYERS:-4}"
EPOCHS="${EPOCHS:-10}"
LOSS="${LOSS:-smoothl1}"
SCHEDULER="${SCHEDULER:-onecycle}"
TARGET_TRANSFORM="${TARGET_TRANSFORM:-standardize}"
DEVICE="${DEVICE:-cuda}"
RUN_PREFIX="${RUN_PREFIX:-phase5}"

for fraction in ${FRACTIONS}; do
    fraction_tag="${fraction//./_}"
    for target in ${TARGETS}; do
        scratch_name="${RUN_PREFIX}_scratch_${target}_tf${fraction_tag}_seed${SEED}"
        finetune_name="${RUN_PREFIX}_finetune_${target}_tf${fraction_tag}_seed${SEED}"

        uv run alignn alignn-train-small \
            --dataset dft_3d \
            --target "${target}" \
            --train-fraction "${fraction}" \
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
            --target-transform "${TARGET_TRANSFORM}" \
            --run-name "${scratch_name}" \
            --device "${DEVICE}"

        uv run alignn alignn-train-small \
            --dataset dft_3d \
            --target "${target}" \
            --train-fraction "${fraction}" \
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
            --target-transform "${TARGET_TRANSFORM}" \
            --pretrained-multitask-checkpoint "${PRETRAINED_CHECKPOINT}" \
            --run-name "${finetune_name}" \
            --device "${DEVICE}"
    done
done
