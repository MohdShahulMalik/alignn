#!/usr/bin/env bash

set -euo pipefail

DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-123}"
BATCH_SIZE="${BATCH_SIZE:-16}"
ALIGNN_LAYERS="${ALIGNN_LAYERS:-4}"
GCN_LAYERS="${GCN_LAYERS:-4}"
LR="${LR:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.00001}"
START_AT="${START_AT:-}"
seen_start="false"

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
)

run_small() {
    local args=("$@")
    local run_name=""
    local i=0
    while [[ $i -lt ${#args[@]} ]]; do
        if [[ "${args[$i]}" == "--run-name" ]] && [[ $((i + 1)) -lt ${#args[@]} ]]; then
            run_name="${args[$((i + 1))]}"
            break
        fi
        i=$((i + 1))
    done

    if [[ -n "${START_AT}" && "${seen_start}" != "true" ]]; then
        if [[ "${run_name}" == "${START_AT}" ]]; then
            seen_start="true"
        else
            echo "[skip] ${run_name:-unknown_run} before START_AT=${START_AT} $(date -Is)"
            return
        fi
    fi

    echo "[start] $* $(date -Is)"
    uv run python -m alignn.cli alignn-train-small "$@"
    echo "[done] $* $(date -Is)"
}

mkdir -p results/tables
if [[ -f "$HOME/projects/2d-alignn/original_alignn_e30_metrics_summary.csv" ]]; then
    cp "$HOME/projects/2d-alignn/original_alignn_e30_metrics_summary.csv" \
        results/tables/original_alignn_e30_metrics_summary_frozen.csv
fi

# Formation energy: positive-tail-aware recipe from the prior report, plus a
# stronger high-positive variant. Goal: beat original 30-epoch MAE/RMSE.
run_small \
    --target formation_energy_peratom \
    "${common[@]}" \
    --hidden-dim 128 \
    --epochs 60 \
    --loss smoothl1 \
    --target-transform none \
    --positive-weight 2.0 \
    --high-target-threshold 1.0 \
    --high-positive-weight 4.0 \
    --mse-tail-weight 0.1 \
    --run-name improve_scratch_formation_energy_h128_tail_e60_seed123

run_small \
    --target formation_energy_peratom \
    "${common[@]}" \
    --hidden-dim 128 \
    --epochs 60 \
    --loss smoothl1 \
    --target-transform none \
    --positive-weight 2.5 \
    --high-target-threshold 0.75 \
    --high-positive-weight 6.0 \
    --mse-tail-weight 0.2 \
    --run-name improve_scratch_formation_energy_h128_tail_stronger_e60_seed123

# Ehull: near-zero-focused recipe from prior report.
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
    --run-name improve_scratch_ehull_h128_log1p_low6_downhigh_e30_seed123

# optB88vdW band gap: metallic plus high-gap weighting from prior report.
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
    --run-name improve_scratch_optb88vdw_bandgap_h128_low2_high4_clamp_e35_seed123

# MBJ band gap: same zero-inflated family, with both robust and MSE variants.
run_small \
    --target mbj_bandgap \
    "${common[@]}" \
    --hidden-dim 128 \
    --epochs 50 \
    --loss smoothl1 \
    --target-transform none \
    --low-target-threshold 0.05 \
    --low-target-weight 2.0 \
    --high-target-threshold 1.0 \
    --high-positive-weight 4.0 \
    --prediction-min 0 \
    --run-name improve_scratch_mbj_bandgap_h128_low2_high4_clamp_e50_seed123

run_small \
    --target mbj_bandgap \
    "${common[@]}" \
    --hidden-dim 128 \
    --epochs 50 \
    --loss mse \
    --target-transform none \
    --prediction-min 0 \
    --run-name improve_scratch_mbj_bandgap_h128_mse_clamp_e50_seed123

# Bulk modulus: high-modulus-tail variants. The 30-epoch original is the hard
# baseline, so include higher capacity and longer schedules.
run_small \
    --target bulk_modulus_kv \
    "${common[@]}" \
    --hidden-dim 128 \
    --epochs 60 \
    --loss smoothl1 \
    --target-transform none \
    --low-target-threshold 10 \
    --low-target-weight 0.8 \
    --high-target-threshold 200 \
    --high-positive-weight 2.0 \
    --prediction-min 0 \
    --run-name improve_scratch_bulk_modulus_kv_h128_low08_high2_clamp_e60_seed123

run_small \
    --target bulk_modulus_kv \
    "${common[@]}" \
    --hidden-dim 256 \
    --epochs 60 \
    --loss smoothl1 \
    --target-transform none \
    --low-target-threshold 10 \
    --low-target-weight 0.8 \
    --high-target-threshold 200 \
    --high-positive-weight 3.0 \
    --prediction-min 0 \
    --run-name improve_scratch_bulk_modulus_kv_h256_low08_high3_clamp_e60_seed123

run_small \
    --target bulk_modulus_kv \
    "${common[@]}" \
    --hidden-dim 128 \
    --epochs 60 \
    --loss mse \
    --target-transform none \
    --high-target-threshold 200 \
    --high-positive-weight 2.0 \
    --prediction-min 0 \
    --selection-metric rmse \
    --run-name improve_scratch_bulk_modulus_kv_h128_mse_high2_rmse_clamp_e60_seed123

uv run python scripts/analyze_multitask_results.py
echo "[all done] $(date -Is)"
