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

log_file="${LOGDIR}/multitask_fix_sweep.log"
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

# Config 1: Equal weighting (baseline with bug fix)
run_configs "mt_equal_w_tf1_0_seed123_e30"

# Config 2: Per-target weighting (upweight weak targets)
run_configs "mt_weighted_w_tf1_0_seed123_e30" \
    --target-weights "bulk_modulus_kv:2.0,formation_energy_peratom:1.5"

# Config 3: Gradient surgery (PCGrad)
run_configs "mt_pcgrad_tf1_0_seed123_e30" \
    --gradient-surgery

# Config 4: Uncertainty weighting (Kendall)
run_configs "mt_uncertain_tf1_0_seed123_e30" \
    --uncertainty-weighting

# Config 5: Weighted + PCGrad
run_configs "mt_weighted_pcgrad_tf1_0_seed123_e30" \
    --target-weights "bulk_modulus_kv:2.0,formation_energy_peratom:1.5" \
    --gradient-surgery

# Config 6: Weighted + Uncertainty
run_configs "mt_weighted_uncertain_tf1_0_seed123_e30" \
    --target-weights "bulk_modulus_kv:2.0,formation_energy_peratom:1.5" \
    --uncertainty-weighting

# Config 7: Stronger weighting for bulk
run_configs "mt_bulk3x_tf1_0_seed123_e30" \
    --target-weights "bulk_modulus_kv:3.0,formation_energy_peratom:2.0"

# Config 8: PCGrad + Uncertainty
run_configs "mt_pcgrad_uncertain_tf1_0_seed123_e30" \
    --gradient-surgery \
    --uncertainty-weighting

echo "[sweep done] $(date -Iseconds)" >> "$log_file"
