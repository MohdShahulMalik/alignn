from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TARGETS = [
    "formation_energy_peratom",
    "ehull",
    "optb88vdw_bandgap",
    "mbj_bandgap",
    "bulk_modulus_kv",
]


def _metric_rows(logs_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(logs_dir.glob("*_test_metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        run_name = path.name.removesuffix("_test_metrics.json")
        metrics = payload.get("test_metrics")
        if not isinstance(metrics, dict):
            continue
        if "mae" in metrics:
            target = payload.get("target") or payload.get("target_column")
            if target:
                rows.append(
                    {
                        "target": target,
                        "run_name": run_name,
                        "mae": metrics.get("mae"),
                        "rmse": metrics.get("rmse"),
                        "p95_abs_error": metrics.get("p95_abs_error"),
                    }
                )
        else:
            for target, target_metrics in metrics.items():
                rows.append(
                    {
                        "target": target,
                        "run_name": run_name,
                        "mae": target_metrics.get("mae"),
                        "rmse": target_metrics.get("rmse"),
                        "p95_abs_error": target_metrics.get("p95_abs_error"),
                    }
                )
    return rows


def _method_group(run_name: str) -> str:
    if run_name.startswith("improve_scratch_"):
        return "improved_scratch"
    if run_name.startswith("strongbaseline_scratch_"):
        return "scratch_e30"
    if "finetune" in run_name:
        return "finetune"
    if "joint" in run_name or "group_ablation" in run_name or "pretrain" in run_name:
        return "joint_or_group"
    if run_name.startswith("dft3d_"):
        return "prior_tuned"
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare reimplementation metrics against frozen original ALIGNN E30 numbers."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--original-summary",
        type=Path,
        default=Path.home() / "projects/2d-alignn/original_alignn_e30_metrics_summary.csv",
    )
    parser.add_argument("--logs-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    logs_dir = args.logs_dir or project_root / "results" / "logs"
    output_dir = args.output_dir or project_root / "results" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    original = pd.read_csv(args.original_summary)
    original = original.rename(
        columns={
            "mae": "original_mae",
            "rmse": "original_rmse",
            "p95_abs_error": "original_p95_abs_error",
        }
    )
    original = original[
        ["target", "original_mae", "original_rmse", "original_p95_abs_error"]
    ]
    original.to_csv(output_dir / "original_alignn_e30_metrics_summary_frozen.csv", index=False)

    metrics = pd.DataFrame(_metric_rows(logs_dir))
    metrics = metrics[metrics["target"].isin(TARGETS)].dropna(subset=["mae"])
    metrics["method_group"] = metrics["run_name"].map(_method_group)
    metrics = metrics.merge(original, on="target", how="left")
    metrics["beats_original_mae"] = metrics["mae"] < metrics["original_mae"]
    metrics["beats_original_rmse"] = metrics["rmse"] < metrics["original_rmse"]
    metrics["mae_delta_vs_original"] = metrics["mae"] - metrics["original_mae"]
    metrics["rmse_delta_vs_original"] = metrics["rmse"] - metrics["original_rmse"]

    metrics.sort_values(["target", "mae", "rmse"]).to_csv(
        output_dir / "e30_original_vs_all_reimplementation_runs.csv",
        index=False,
    )

    best_overall = metrics.loc[metrics.groupby("target")["mae"].idxmin()].copy()
    best_scratch = metrics[metrics["method_group"].isin(["improved_scratch", "scratch_e30", "prior_tuned"])]
    best_scratch = best_scratch.loc[best_scratch.groupby("target")["mae"].idxmin()].copy()

    best_overall["selection"] = "best_overall"
    best_scratch["selection"] = "best_scratch_like"
    summary = pd.concat([best_overall, best_scratch], ignore_index=True)
    summary.sort_values(["target", "selection"]).to_csv(
        output_dir / "e30_original_vs_best_reimplementation_summary.csv",
        index=False,
    )

    print(f"all_runs={output_dir / 'e30_original_vs_all_reimplementation_runs.csv'}")
    print(f"summary={output_dir / 'e30_original_vs_best_reimplementation_summary.csv'}")
    print(
        summary[
            [
                "selection",
                "target",
                "run_name",
                "mae",
                "original_mae",
                "beats_original_mae",
                "rmse",
                "original_rmse",
                "beats_original_rmse",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
