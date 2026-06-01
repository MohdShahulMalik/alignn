#!/usr/bin/env bash

set -euo pipefail

DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-123}"
BATCH_SIZE="${BATCH_SIZE:-16}"
ALIGNN_LAYERS="${ALIGNN_LAYERS:-4}"
GCN_LAYERS="${GCN_LAYERS:-4}"
LR="${LR:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.00001}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

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
    --use-cudnn-benchmark
)

fast_common=("${common[@]}")

run_small() {
    echo "[start] $* $(date -Is)"
    "${PYTHON_BIN}" -m alignn.cli alignn-train-small "$@"
    echo "[done] $* $(date -Is)"
}

# Formation energy variants
run_small \
    --target formation_energy_peratom \
    "${fast_common[@]}" \
    --hidden-dim 64 \
    --epochs 60 \
    --loss mse \
    --target-transform none \
    --run-name fast_formation_energy_h64_mse_e60_seed123

run_small \
    --target formation_energy_peratom \
    "${fast_common[@]}" \
    --hidden-dim 128 \
    --epochs 60 \
    --loss mse \
    --target-transform none \
    --run-name fast_formation_energy_h128_mse_e60_seed123

run_small \
    --target formation_energy_peratom \
    "${fast_common[@]}" \
    --hidden-dim 64 \
    --epochs 60 \
    --loss mse \
    --target-transform none \
    --readout meanmax \
    --run-name fast_formation_energy_h64_mse_meanmax_e60_seed123

run_small \
    --target formation_energy_peratom \
    "${fast_common[@]}" \
    --hidden-dim 128 \
    --epochs 60 \
    --loss mse \
    --target-transform none \
    --readout meanmax \
    --run-name fast_formation_energy_h128_mse_meanmax_e60_seed123

# Bulk modulus variants with model-level penalty
run_small \
    --target bulk_modulus_kv \
    "${fast_common[@]}" \
    --hidden-dim 64 \
    --epochs 60 \
    --loss mse \
    --target-transform none \
    --energy-mult-natoms \
    --penalty-factor 0.1 \
    --penalty-threshold 1.0 \
    --selection-metric rmse \
    --run-name fast_bulk_modulus_kv_h64_mse_scaled_penalty_rmse_e60_seed123

run_small \
    --target bulk_modulus_kv \
    "${fast_common[@]}" \
    --hidden-dim 128 \
    --epochs 60 \
    --loss smoothl1 \
    --target-transform none \
    --energy-mult-natoms \
    --penalty-factor 0.1 \
    --penalty-threshold 1.0 \
    --prediction-min 0 \
    --run-name fast_bulk_modulus_kv_h128_smoothl1_scaled_penalty_e60_seed123

# Regression checks - already-winning recipes
run_small \
    --target ehull \
    "${fast_common[@]}" \
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
    --run-name fast_ehull_h128_log1p_low6_downhigh_e30_seed123

run_small \
    --target optb88vdw_bandgap \
    "${fast_common[@]}" \
    --hidden-dim 128 \
    --epochs 35 \
    --loss smoothl1 \
    --target-transform none \
    --low-target-threshold 0.05 \
    --low-target-weight 2.0 \
    --high-target-threshold 1.0 \
    --high-positive-weight 4.0 \
    --prediction-min 0 \
    --run-name fast_optb88vdw_bandgap_h128_low2_high4_clamp_e35_seed123

run_small \
    --target mbj_bandgap \
    "${fast_common[@]}" \
    --hidden-dim 128 \
    --epochs 50 \
    --loss mse \
    --target-transform none \
    --prediction-min 0 \
    --run-name fast_mbj_bandgap_h128_mse_clamp_e50_seed123

run_small \
    --target mbj_bandgap \
    "${fast_common[@]}" \
    --hidden-dim 128 \
    --epochs 50 \
    --loss smoothl1 \
    --target-transform none \
    --low-target-threshold 0.05 \
    --low-target-weight 2.0 \
    --high-target-threshold 1.0 \
    --high-positive-weight 4.0 \
    --prediction-min 0 \
    --run-name fast_mbj_bandgap_h128_low2_high4_clamp_e50_seed123

"${PYTHON_BIN}" scripts/compare_against_original_e30.py \
    --original-summary "$HOME/projects/2d-alignn/original_alignn_e60_metrics_summary.csv" \
    --output-dir results/tables/e60_fast \
    --comparison-label e60_fast

"${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import json
import pandas as pd

run_names = [
    "fast_formation_energy_h64_mse_e60_seed123",
    "fast_formation_energy_h128_mse_e60_seed123",
    "fast_formation_energy_h64_mse_meanmax_e60_seed123",
    "fast_formation_energy_h128_mse_meanmax_e60_seed123",
    "fast_bulk_modulus_kv_h64_mse_scaled_penalty_rmse_e60_seed123",
    "fast_bulk_modulus_kv_h128_smoothl1_scaled_penalty_e60_seed123",
    "fast_ehull_h128_log1p_low6_downhigh_e30_seed123",
    "fast_optb88vdw_bandgap_h128_low2_high4_clamp_e35_seed123",
    "fast_mbj_bandgap_h128_mse_clamp_e50_seed123",
    "fast_mbj_bandgap_h128_low2_high4_clamp_e50_seed123",
]
original = pd.read_csv(Path.home() / "projects/2d-alignn/original_alignn_e60_metrics_summary.csv")
rows = []
for run_name in run_names:
    path = Path("results/logs") / f"{run_name}_test_metrics.json"
    if not path.exists():
        rows.append({"run_name": run_name, "status": "missing"})
        continue
    payload = json.loads(path.read_text(encoding="utf-8"))
    target = payload["target"]
    metrics = payload["test_metrics"]
    orig = original[original["target"].eq(target)].iloc[0]
    rows.append(
        {
            "run_name": run_name,
            "target": target,
            "status": "complete",
            "mae": metrics["mae"],
            "original_mae": orig["mae"],
            "mae_delta_vs_original": metrics["mae"] - orig["mae"],
            "beats_original_mae": metrics["mae"] < orig["mae"],
            "rmse": metrics["rmse"],
            "original_rmse": orig["rmse"],
            "rmse_delta_vs_original": metrics["rmse"] - orig["rmse"],
            "beats_original_rmse": metrics["rmse"] < orig["rmse"],
            "p95_abs_error": metrics["p95_abs_error"],
            "original_p95_abs_error": orig["p95_abs_error"],
        }
    )

output = Path("results/tables/e60_fast/fast_target_check_summary.csv")
output.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(output, index=False)
print(f"fast_summary={output}")
print(pd.DataFrame(rows).to_string(index=False))
PY

echo "[all done] $(date -Is)"
