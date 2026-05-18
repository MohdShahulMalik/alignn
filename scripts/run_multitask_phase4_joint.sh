#!/usr/bin/env bash

set -euo pipefail

TARGETS="${TARGETS:-formation_energy_peratom,ehull,optb88vdw_bandgap,mbj_bandgap,bulk_modulus_kv}"
SEED="${SEED:-123}"
TRAIN_FRACTION="${TRAIN_FRACTION:-1.0}"
BATCH_SIZE="${BATCH_SIZE:-16}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
HEAD_HIDDEN_DIM="${HEAD_HIDDEN_DIM:-0}"
ALIGNN_LAYERS="${ALIGNN_LAYERS:-4}"
GCN_LAYERS="${GCN_LAYERS:-4}"
EPOCHS="${EPOCHS:-10}"
LOSS="${LOSS:-smoothl1}"
SCHEDULER="${SCHEDULER:-onecycle}"
DEVICE="${DEVICE:-cuda}"
RUN_PREFIX="${RUN_PREFIX:-phase4_joint}"

fraction_tag="${TRAIN_FRACTION//./_}"
run_name="${RUN_PREFIX}_tf${fraction_tag}_seed${SEED}"
cmd=(
    uv run alignn alignn-train-multitask
    --dataset dft_3d
    --targets "${TARGETS}"
    --train-fraction "${TRAIN_FRACTION}"
    --train-subset-size 0
    --val-subset-size 0
    --test-subset-size 0
    --batch-size "${BATCH_SIZE}"
    --hidden-dim "${HIDDEN_DIM}"
    --alignn-layers "${ALIGNN_LAYERS}"
    --gcn-layers "${GCN_LAYERS}"
    --epochs "${EPOCHS}"
    --seed "${SEED}"
    --loss "${LOSS}"
    --scheduler "${SCHEDULER}"
    --run-name "${run_name}"
    --device "${DEVICE}"
)
if [[ "${HEAD_HIDDEN_DIM}" != "0" ]]; then
    cmd+=(--head-hidden-dim "${HEAD_HIDDEN_DIM}")
fi

"${cmd[@]}"
