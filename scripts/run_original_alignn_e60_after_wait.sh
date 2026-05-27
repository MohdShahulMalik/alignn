#!/usr/bin/env bash

set -euo pipefail

WAIT_PID="${WAIT_PID:-}"
ORIGINAL_ROOT="${ORIGINAL_ROOT:-$HOME/projects/2d-alignn}"
CONDA_ENV="${CONDA_ENV:-2d-alignn}"

if [[ -n "${WAIT_PID}" ]]; then
    while kill -0 "${WAIT_PID}" 2>/dev/null; do
        echo "[wait] process ${WAIT_PID} still running $(date -Is)"
        sleep 300
    done
fi

source "$HOME/conda/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"

cd "${ORIGINAL_ROOT}"

python - <<'PY'
from pathlib import Path
import json

configs = {
    "formation_energy_peratom": (
        "fair_compare_dft3d_formation_energy_peratom_atomwise_e30_config.json",
        "fair_compare_dft3d_formation_energy_peratom_atomwise_e60",
        "fair_dft3d_formation_energy_peratom_atomwise_e60_",
    ),
    "ehull": (
        "fair_compare_dft3d_ehull_atomwise_e30_config.json",
        "fair_compare_dft3d_ehull_atomwise_e60",
        "fair_dft3d_ehull_atomwise_e60_",
    ),
    "optb88vdw_bandgap": (
        "fair_compare_dft3d_optb88vdw_bandgap_atomwise_e30_config.json",
        "fair_compare_dft3d_optb88vdw_bandgap_atomwise_e60",
        "fair_dft3d_optb88vdw_bandgap_atomwise_e60_",
    ),
    "mbj_bandgap": (
        "fair_compare_dft3d_mbj_bandgap_atomwise_e30_config.json",
        "fair_compare_dft3d_mbj_bandgap_atomwise_e60",
        "fair_dft3d_mbj_bandgap_atomwise_e60_",
    ),
    "bulk_modulus_kv": (
        "fair_compare_dft3d_bulk_modulus_kv_atomwise_e30_config.json",
        "fair_compare_dft3d_bulk_modulus_kv_atomwise_e60",
        "fair_dft3d_bulk_modulus_kv_atomwise_e60_",
    ),
}

for target, (source_name, output_dir, filename) in configs.items():
    source_path = Path(source_name)
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    config = json.loads(source_path.read_text(encoding="utf-8"))
    config["target"] = target
    config["epochs"] = 60
    config["output_dir"] = "./" + output_dir
    config["filename"] = filename
    config["write_predictions"] = True
    config["store_outputs"] = True
    config["write_checkpoint"] = True
    config["progress"] = True
    path = Path(f"fair_compare_dft3d_{target}_atomwise_e60_config.json")
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"config={path} output_dir={config['output_dir']}")
PY

configs="${CONFIGS:-fair_compare_dft3d_formation_energy_peratom_atomwise_e60_config.json fair_compare_dft3d_ehull_atomwise_e60_config.json fair_compare_dft3d_optb88vdw_bandgap_atomwise_e60_config.json fair_compare_dft3d_mbj_bandgap_atomwise_e60_config.json fair_compare_dft3d_bulk_modulus_kv_atomwise_e60_config.json}"
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
    ("formation_energy_peratom", "fair_compare_dft3d_formation_energy_peratom_atomwise_e60"),
    ("ehull", "fair_compare_dft3d_ehull_atomwise_e60"),
    ("optb88vdw_bandgap", "fair_compare_dft3d_optb88vdw_bandgap_atomwise_e60"),
    ("mbj_bandgap", "fair_compare_dft3d_mbj_bandgap_atomwise_e60"),
    ("bulk_modulus_kv", "fair_compare_dft3d_bulk_modulus_kv_atomwise_e60"),
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

output_path = Path("original_alignn_e60_metrics_summary.csv")
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
