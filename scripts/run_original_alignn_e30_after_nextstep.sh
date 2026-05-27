#!/usr/bin/env bash

set -euo pipefail

WAIT_PID="${WAIT_PID:-50715}"
ORIGINAL_ROOT="${ORIGINAL_ROOT:-$HOME/projects/2d-alignn}"
CONDA_ENV="${CONDA_ENV:-2d-alignn}"

while kill -0 "${WAIT_PID}" 2>/dev/null; do
    echo "[wait] alignn pipeline ${WAIT_PID} still running $(date -Is)"
    sleep 300
done

source "$HOME/conda/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"

cd "${ORIGINAL_ROOT}"

configs="${CONFIGS:-fair_compare_dft3d_formation_energy_peratom_atomwise_e30_config.json fair_compare_dft3d_ehull_atomwise_e30_config.json fair_compare_dft3d_optb88vdw_bandgap_atomwise_e30_config.json fair_compare_dft3d_mbj_bandgap_atomwise_e30_config.json fair_compare_dft3d_bulk_modulus_kv_atomwise_e30_config.json}"
for config in ${configs}; do
    echo "[start] original ALIGNN ${config} $(date -Is)"
    python -c "import json, sys; from pathlib import Path; from alignn.config import TrainingConfig; from alignn.train import train_dgl; cfg=TrainingConfig(**json.loads(Path(sys.argv[1]).read_text())); train_dgl(cfg)" "${config}"
    echo "[done] original ALIGNN ${config} $(date -Is)"
done

python - <<'PY'
from pathlib import Path
import csv
import math

rows = []
items = [
    ("formation_energy_peratom", "fair_compare_dft3d_formation_energy_peratom_atomwise_e30"),
    ("ehull", "fair_compare_dft3d_ehull_atomwise_e30"),
    ("optb88vdw_bandgap", "fair_compare_dft3d_optb88vdw_bandgap_atomwise_e30"),
    ("mbj_bandgap", "fair_compare_dft3d_mbj_bandgap_atomwise_e30"),
    ("bulk_modulus_kv", "fair_compare_dft3d_bulk_modulus_kv_atomwise_e30"),
]
for target, dirname in items:
    path = Path(dirname) / "prediction_results_test_set.csv"
    if not path.exists():
        rows.append({"target": target, "output_dir": dirname, "status": "missing"})
        continue
    with path.open() as handle:
        data = list(csv.DictReader(handle))
    errors = [float(row["prediction"]) - float(row["target"]) for row in data]
    abs_errors = [abs(error) for error in errors]
    rows.append(
        {
            "target": target,
            "output_dir": dirname,
            "status": "complete",
            "test_size": len(data),
            "mae": sum(abs_errors) / len(abs_errors),
            "rmse": math.sqrt(sum(error * error for error in errors) / len(errors)),
            "p95_abs_error": sorted(abs_errors)[int(math.ceil(0.95 * len(abs_errors))) - 1],
        }
    )

output_path = Path("original_alignn_e30_metrics_summary.csv")
fieldnames = sorted({key for row in rows for key in row})
with output_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(f"summary={output_path}")
for row in rows:
    print(row)
PY

echo "[all done] $(date -Is)"
