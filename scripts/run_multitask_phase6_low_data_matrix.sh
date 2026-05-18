#!/usr/bin/env bash

set -euo pipefail

TARGETS_CSV="${TARGETS_CSV:-formation_energy_peratom,ehull,optb88vdw_bandgap,mbj_bandgap,bulk_modulus_kv}"
TARGETS="${TARGETS:-formation_energy_peratom ehull optb88vdw_bandgap mbj_bandgap bulk_modulus_kv}"
FRACTIONS="${FRACTIONS:-0.10 0.25 0.50 1.0}"
SEEDS="${SEEDS:-123 234 345}"
RUN_PREFIX="${RUN_PREFIX:-phase6}"

for seed in ${SEEDS}; do
    SEED="${seed}" TARGETS="${TARGETS_CSV}" RUN_PREFIX="${RUN_PREFIX}_pretrain" \
        bash scripts/run_multitask_phase4_joint.sh
    pretrained="results/checkpoints/${RUN_PREFIX}_pretrain_tf1_0_seed${seed}.pt"

    for fraction in ${FRACTIONS}; do
        SEED="${seed}" TARGETS="${TARGETS_CSV}" TRAIN_FRACTION="${fraction}" RUN_PREFIX="${RUN_PREFIX}_joint" \
            bash scripts/run_multitask_phase4_joint.sh
    done

    SEED="${seed}" TARGETS="${TARGETS}" FRACTIONS="${FRACTIONS}" \
        PRETRAINED_CHECKPOINT="${pretrained}" RUN_PREFIX="${RUN_PREFIX}" \
        bash scripts/run_multitask_phase5_finetune.sh
done
