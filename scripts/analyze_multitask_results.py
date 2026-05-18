from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _infer_model_type(run_name: str) -> str:
    if "finetune" in run_name:
        return "pretrained_finetune"
    if "scratch" in run_name or "single" in run_name:
        return "single_task"
    if "joint" in run_name or "pretrain" in run_name or "multitask" in run_name:
        return "multi_task_joint"
    return "unknown"


def _metric_rows(metrics_path: Path) -> list[dict[str, Any]]:
    payload = _read_json(metrics_path)
    run_name = metrics_path.name.removesuffix("_test_metrics.json")
    base = {
        "run_name": run_name,
        "model_type": _infer_model_type(run_name),
        "metrics_path": str(metrics_path),
        "best_val_standardized_mae": payload.get("best_val_standardized_mae"),
        "best_val_score": payload.get("best_val_score"),
    }
    rows: list[dict[str, Any]] = []
    if "test_metrics" in payload and isinstance(payload["test_metrics"], dict):
        if "mae" in payload["test_metrics"]:
            target = payload.get("target") or payload.get("target_column")
            rows.append({**base, "target": target, **payload["test_metrics"]})
        else:
            best_epochs = payload.get("best_epoch_by_target", {})
            sizes = payload.get("sizes", {})
            for target, metrics in payload["test_metrics"].items():
                rows.append(
                    {
                        **base,
                        "target": target,
                        "best_epoch": best_epochs.get(target),
                        "train_size": sizes.get(target, {}).get("train"),
                        "val_size": sizes.get(target, {}).get("val"),
                        "test_size": sizes.get(target, {}).get("test"),
                        **metrics,
                    }
                )
    return rows


def _write_metric_summary(logs_dir: Path, output_dir: Path) -> Path:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(logs_dir.glob("*_test_metrics.json")):
        rows.extend(_metric_rows(metrics_path))
    output_path = output_dir / "multitask_metrics_summary.csv"
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return output_path
    fieldnames = sorted({key for row in rows for key in row})
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def _prediction_files(logs_dir: Path) -> dict[str, Path]:
    return {
        path.name.removesuffix("_test_predictions.csv"): path
        for path in logs_dir.glob("*_test_predictions.csv")
    }


def _load_predictions(path: Path, default_target: str | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "target_name" not in frame.columns:
        frame["target_name"] = default_target or "unknown"
    return frame[["jid", "target_name", "target", "prediction", "abs_error"]].copy()


def _write_pairwise_transfer(
    logs_dir: Path,
    output_dir: Path,
    baseline_run: str | None,
    candidate_run: str | None,
) -> Path | None:
    if not baseline_run or not candidate_run:
        return None
    files = _prediction_files(logs_dir)
    if baseline_run not in files:
        raise FileNotFoundError(f"Missing baseline predictions for run: {baseline_run}")
    if candidate_run not in files:
        raise FileNotFoundError(f"Missing candidate predictions for run: {candidate_run}")
    baseline = _load_predictions(files[baseline_run]).rename(
        columns={"prediction": "baseline_prediction", "abs_error": "baseline_abs_error"}
    )
    candidate = _load_predictions(files[candidate_run]).rename(
        columns={"prediction": "candidate_prediction", "abs_error": "candidate_abs_error"}
    )
    merged = baseline.merge(
        candidate,
        on=["jid", "target_name", "target"],
        how="inner",
    )
    merged["abs_error_delta"] = merged["baseline_abs_error"] - merged["candidate_abs_error"]
    merged["candidate_wins"] = merged["abs_error_delta"] > 0
    output_path = output_dir / f"{candidate_run}_vs_{baseline_run}_per_sample.csv"
    merged.to_csv(output_path, index=False)

    grouped = (
        merged.groupby("target_name")
        .agg(
            n=("jid", "count"),
            mean_abs_error_delta=("abs_error_delta", "mean"),
            win_rate=("candidate_wins", "mean"),
        )
        .reset_index()
    )
    grouped.to_csv(
        output_dir / f"{candidate_run}_vs_{baseline_run}_target_summary.csv",
        index=False,
    )
    return output_path


def _write_target_correlations(project_root: Path, targets: list[str], output_dir: Path) -> Path:
    split_dir = project_root / "data" / "splits"
    frames = []
    for target in targets:
        path = split_dir / f"dft_3d_{target}_train.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)[["jid", "target"]].rename(columns={"target": target})
        frames.append(frame)
    output_path = output_dir / "multitask_target_correlations.csv"
    if not frames:
        output_path.write_text("", encoding="utf-8")
        return output_path
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="jid", how="inner")
    corr = merged.drop(columns=["jid"]).corr(method="pearson")
    corr.to_csv(output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate multi-task ALIGNN results.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--logs-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--baseline-run", default=None)
    parser.add_argument("--candidate-run", default=None)
    parser.add_argument(
        "--targets",
        default="formation_energy_peratom,ehull,optb88vdw_bandgap,mbj_bandgap,bulk_modulus_kv",
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    logs_dir = args.logs_dir or project_root / "results" / "logs"
    output_dir = args.output_dir or project_root / "results" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = _write_metric_summary(logs_dir=logs_dir, output_dir=output_dir)
    pairwise_path = _write_pairwise_transfer(
        logs_dir=logs_dir,
        output_dir=output_dir,
        baseline_run=args.baseline_run,
        candidate_run=args.candidate_run,
    )
    correlation_path = _write_target_correlations(
        project_root=project_root,
        targets=[target.strip() for target in args.targets.split(",") if target.strip()],
        output_dir=output_dir,
    )
    print(f"metrics_summary={metrics_path}")
    if pairwise_path is not None:
        print(f"pairwise_transfer={pairwise_path}")
    print(f"target_correlations={correlation_path}")


if __name__ == "__main__":
    main()
