#!/usr/bin/env bash
set -euo pipefail

VENV_PYTHON="$HOME/projects/alignn/.venv/bin/python"
export LD_LIBRARY_PATH="$HOME/projects/alignn/.venv/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:$HOME/projects/alignn/.venv/lib/python3.11/site-packages/nvidia/cudnn/lib:$HOME/projects/alignn/.venv/lib/python3.11/site-packages/nvidia/cufft/lib:$HOME/projects/alignn/.venv/lib/python3.11/site-packages/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

LOGDIR="$HOME/projects/alignn/results/logs"
mkdir -p "$LOGDIR"

PROJECT_ROOT="$HOME/projects/alignn"
CMD="alignn-train-multitask"
TARGETS="formation_energy_peratom,ehull,optb88vdw_bandgap,mbj_bandgap,bulk_modulus_kv"

common_args=(
    --dataset dft_3d
    --train-fraction 1.0
    --alignn-layers 4
    --gcn-layers 4
    --seed 123
    --learning-rate 0.001
    --weight-decay 0.00001
    --scheduler onecycle
    --device cuda
    --hidden-dim 128
    --batch-size 16
    --epochs 30
    --loss l1
    --num-workers 0
)

log_file="${LOGDIR}/multitask_final_sweep.log"

run_configs() {
    local run_name="$1"; shift
    local extra_args=("$@")
    echo "[start] $run_name $(date -Iseconds)" >> "$log_file"
    $VENV_PYTHON -m alignn.cli "$CMD" \
        --targets "$TARGETS" \
        --run-name "$run_name" \
        --project-root "$PROJECT_ROOT" \
        "${common_args[@]}" \
        "${extra_args[@]}" \
        >> "$LOGDIR/${run_name}.log" 2>&1
    echo "[done] $run_name $(date -Iseconds)" >> "$log_file"
}

# Config 7: DB-MTL + bulk_kv head + selective PCGrad
run_configs "mt_final_bulkhead_pcgrad" \
    --dual-balancing --gradient-surgery \
    --target-weights "bulk_modulus_kv:2.0,formation_energy_peratom:1.5" \
    --head-configs '{"bulk_modulus_kv": {"hidden_dim": 128, "n_layers": 3}}' \
    --selective-pcgrad-targets "bulk_modulus_kv"

# Config 8: DB-MTL + all combined
run_configs "mt_final_all_combined" \
    --dual-balancing --gradient-surgery \
    --target-weights "bulk_modulus_kv:2.0,formation_energy_peratom:1.5" \
    --head-configs '{"bulk_modulus_kv": {"hidden_dim": 128, "n_layers": 3, "dropout": 0.15}}' \
    --head-dropout 0.1 \
    --selective-pcgrad-targets "bulk_modulus_kv"

echo "[sweep-cont done] $(date -Iseconds)" >> "$log_file"
