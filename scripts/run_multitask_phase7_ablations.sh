#!/usr/bin/env bash

set -euo pipefail

SEED="${SEED:-123}"
TRAIN_FRACTION="${TRAIN_FRACTION:-1.0}"
ABLATION_GROUPS="${ABLATION_GROUPS:-all thermodynamic electronic mechanical_excluded bandgap}"
RUN_PREFIX="${RUN_PREFIX:-phase7_ablation}"

for group in ${ABLATION_GROUPS}; do
    case "${group}" in
        all)
            targets="formation_energy_peratom,ehull,optb88vdw_bandgap,mbj_bandgap,bulk_modulus_kv"
            ;;
        thermodynamic)
            targets="formation_energy_peratom,ehull"
            ;;
        electronic)
            targets="optb88vdw_bandgap,mbj_bandgap"
            ;;
        mechanical_excluded)
            targets="formation_energy_peratom,ehull,optb88vdw_bandgap,mbj_bandgap"
            ;;
        bandgap)
            targets="optb88vdw_bandgap,mbj_bandgap"
            ;;
        formation_energy_*)
            downstream="${group#formation_energy_}"
            targets="formation_energy_peratom,${downstream}"
            ;;
        ehull_*)
            downstream="${group#ehull_}"
            targets="ehull,${downstream}"
            ;;
        *)
            echo "Unknown ablation group: ${group}" >&2
            exit 1
            ;;
    esac

    SEED="${SEED}" TRAIN_FRACTION="${TRAIN_FRACTION}" TARGETS="${targets}" RUN_PREFIX="${RUN_PREFIX}_${group}" \
        bash scripts/run_multitask_phase4_joint.sh
done
