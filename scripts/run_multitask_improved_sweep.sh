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

log_file="${LOGDIR}/multitask_improved_sweep.log"
echo "[sweep start] $(date -Iseconds)" >> "$log_file"

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

# Config 1: DB-MTL (log-loss + max-norm gradient normalization)
run_configs "mt_dbmtl_tf1_0_seed123_e30" \
    --dual-balancing

# Config 2: DB-MTL + PCGrad
run_configs "mt_dbmtl_pcgrad_tf1_0_seed123_e30" \
    --dual-balancing --gradient-surgery

# Config 3: DB-MTL + Uncertainty weighting
run_configs "mt_dbmtl_uncertain_tf1_0_seed123_e30" \
    --dual-balancing --uncertainty-weighting

# Config 4: DB-MTL + PCGrad + Uncertainty (triple combo)
run_configs "mt_dbmtl_pcgrad_uncertain_tf1_0_seed123_e30" \
    --dual-balancing --gradient-surgery --uncertainty-weighting

# Config 5: GradNorm (alpha=1.5)
run_configs "mt_gradnorm_a15_tf1_0_seed123_e30" \
    --gradnorm --gradnorm-alpha 1.5

# Config 6: GradNorm + PCGrad
run_configs "mt_gradnorm_a15_pcgrad_tf1_0_seed123_e30" \
    --gradnorm --gradnorm-alpha 1.5 --gradient-surgery

# Config 7: DB-MTL + target weights (upweight weak targets)
run_configs "mt_dbmtl_weighted_tf1_0_seed123_e30" \
    --dual-balancing \
    --target-weights "bulk_modulus_kv:2.0,formation_energy_peratom:1.5"

# Config 8: DB-MTL + PCGrad + target weights (best combo from sota)
run_configs "mt_dbmtl_pcgrad_weighted_tf1_0_seed123_e30" \
    --dual-balancing --gradient-surgery \
    --target-weights "bulk_modulus_kv:2.0,formation_energy_peratom:1.5"

echo "[sweep done] $(date -Iseconds)" >> "$log_file"
