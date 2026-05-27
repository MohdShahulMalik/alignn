from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


TARGETS = [
    "formation_energy_peratom",
    "ehull",
    "optb88vdw_bandgap",
    "mbj_bandgap",
    "bulk_modulus_kv",
]
FRACTIONS = [0.10, 0.25, 0.50, 1.0]
SEEDS = [123, 234, 345]


def _fraction_tag(fraction: float) -> str:
    return {
        0.10: "0_10",
        0.25: "0_25",
        0.50: "0_50",
        1.0: "1_0",
    }[fraction]


def _parse_phase6_run(run_name: str) -> dict[str, object] | None:
    match = re.match(r"rest_phase6_(joint)_tf(\d+_\d+)_seed(\d+)$", run_name)
    if match:
        return {
            "method": "joint",
            "fraction": float(match.group(2).replace("_", ".")),
            "seed": int(match.group(3)),
            "target": None,
        }
    match = re.match(
        r"rest_phase6_(scratch|finetune)_(.+)_tf(\d+_\d+)_seed(\d+)$",
        run_name,
    )
    if match:
        return {
            "method": match.group(1),
            "target": match.group(2),
            "fraction": float(match.group(3).replace("_", ".")),
            "seed": int(match.group(4)),
        }
    return None


def _prediction_files(logs_dir: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in logs_dir.glob("rest_phase6*_test_predictions.csv"):
        run_name = path.name.removesuffix("_test_predictions.csv")
        if _parse_phase6_run(run_name) is not None:
            files[run_name] = path
    return files


def _split_metadata(project_root: Path, targets: list[str]) -> dict[str, pd.DataFrame]:
    split_dir = project_root / "data" / "splits"
    metadata: dict[str, pd.DataFrame] = {}
    for target in targets:
        path = split_dir / f"dft_3d_{target}_test.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        columns = ["jid"]
        if "num_atoms" in frame.columns:
            columns.append("num_atoms")
        metadata[target] = frame[columns].drop_duplicates("jid")
    return metadata


def _load_predictions(path: Path, target_name: str | None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "target_name" in frame.columns and target_name is not None:
        frame = frame[frame["target_name"] == target_name]
    elif "target_name" not in frame.columns:
        frame["target_name"] = target_name or "unknown"
    return frame[["jid", "target", "prediction", "abs_error", "target_name"]].copy()


def _write_per_sample_transfer(
    project_root: Path,
    logs_dir: Path,
    output_dir: Path,
    targets: list[str],
    fractions: list[float],
    seeds: list[int],
) -> pd.DataFrame:
    files = _prediction_files(logs_dir)
    split_metadata = _split_metadata(project_root, targets)
    rows: list[dict[str, object]] = []
    per_sample_frames: list[pd.DataFrame] = []

    for seed in seeds:
        for fraction in fractions:
            tag = _fraction_tag(fraction)
            joint_run = f"rest_phase6_joint_tf{tag}_seed{seed}"
            for target in targets:
                scratch_run = f"rest_phase6_scratch_{target}_tf{tag}_seed{seed}"
                finetune_run = f"rest_phase6_finetune_{target}_tf{tag}_seed{seed}"
                comparisons = []
                if scratch_run in files and finetune_run in files:
                    comparisons.append(("finetune_vs_scratch", finetune_run, scratch_run))
                if scratch_run in files and joint_run in files:
                    comparisons.append(("joint_vs_scratch", joint_run, scratch_run))

                for comparison, candidate_run, baseline_run in comparisons:
                    candidate = _load_predictions(files[candidate_run], target).rename(
                        columns={
                            "prediction": "candidate_prediction",
                            "abs_error": "candidate_abs_error",
                        }
                    )
                    baseline = _load_predictions(files[baseline_run], target).rename(
                        columns={
                            "prediction": "baseline_prediction",
                            "abs_error": "baseline_abs_error",
                        }
                    )
                    merged = baseline.merge(
                        candidate,
                        on="jid",
                        suffixes=("_baseline", "_candidate"),
                        how="inner",
                    )
                    if merged.empty:
                        continue
                    merged["target_name"] = target
                    merged["target"] = merged["target_baseline"]
                    merged["seed"] = seed
                    merged["fraction"] = fraction
                    merged["comparison"] = comparison
                    merged["baseline_run"] = baseline_run
                    merged["candidate_run"] = candidate_run
                    merged["abs_error_delta"] = (
                        merged["baseline_abs_error"] - merged["candidate_abs_error"]
                    )
                    merged["candidate_wins"] = merged["abs_error_delta"] > 0
                    if target in split_metadata:
                        merged = merged.merge(split_metadata[target], on="jid", how="left")
                    else:
                        merged["num_atoms"] = pd.NA
                    per_sample_frames.append(
                        merged[
                            [
                                "comparison",
                                "seed",
                                "fraction",
                                "target_name",
                                "jid",
                                "target",
                                "baseline_run",
                                "candidate_run",
                                "baseline_prediction",
                                "candidate_prediction",
                                "baseline_abs_error",
                                "candidate_abs_error",
                                "abs_error_delta",
                                "candidate_wins",
                                "num_atoms",
                            ]
                        ]
                    )
                    rows.append(
                        {
                            "comparison": comparison,
                            "seed": seed,
                            "fraction": fraction,
                            "target": target,
                            "n": len(merged),
                            "baseline_mae": merged["baseline_abs_error"].mean(),
                            "candidate_mae": merged["candidate_abs_error"].mean(),
                            "mean_abs_error_delta": merged["abs_error_delta"].mean(),
                            "median_abs_error_delta": merged["abs_error_delta"].median(),
                            "win_rate": merged["candidate_wins"].mean(),
                        }
                    )

    per_sample = pd.concat(per_sample_frames, ignore_index=True)
    summary = pd.DataFrame(rows)
    per_sample.to_csv(output_dir / "phase6_per_sample_transfer.csv", index=False)
    summary.to_csv(output_dir / "phase6_per_sample_transfer_summary.csv", index=False)
    return per_sample


def _write_regime_summaries(per_sample: pd.DataFrame, output_dir: Path) -> None:
    summary = (
        per_sample.groupby(["comparison", "fraction", "target_name"])
        .agg(
            n=("jid", "count"),
            baseline_mae=("baseline_abs_error", "mean"),
            candidate_mae=("candidate_abs_error", "mean"),
            mean_abs_error_delta=("abs_error_delta", "mean"),
            win_rate=("candidate_wins", "mean"),
        )
        .reset_index()
        .rename(columns={"target_name": "target"})
    )
    summary.to_csv(output_dir / "phase6_transfer_by_target_fraction.csv", index=False)

    regime = per_sample.copy()
    regime["target_bin"] = regime.groupby(["comparison", "target_name"])[
        "target"
    ].transform(lambda s: pd.qcut(s, q=4, duplicates="drop", labels=False))
    regime["num_atoms_bin"] = regime.groupby(["comparison", "target_name"])[
        "num_atoms"
    ].transform(
        lambda s: pd.qcut(
            s.rank(method="first"),
            q=4,
            labels=["small", "medium_small", "medium_large", "large"],
        )
    )

    bandgap_targets = {"optb88vdw_bandgap", "mbj_bandgap"}
    bandgap_mask = regime["target_name"].isin(bandgap_targets)
    regime.loc[bandgap_mask, "bandgap_class"] = regime.loc[
        bandgap_mask,
        "target",
    ].map(lambda value: "zero_gap" if abs(float(value)) < 1e-12 else "positive_gap")
    regime.loc[~bandgap_mask, "bandgap_class"] = "not_bandgap"

    target_bins = (
        regime.groupby(["comparison", "target_name", "target_bin"])
        .agg(
            n=("jid", "count"),
            mean_target=("target", "mean"),
            mean_abs_error_delta=("abs_error_delta", "mean"),
            win_rate=("candidate_wins", "mean"),
        )
        .reset_index()
    )
    target_bins.to_csv(output_dir / "phase6_transfer_by_target_value_bin.csv", index=False)

    atom_bins = (
        regime.groupby(["comparison", "target_name", "num_atoms_bin"], observed=True)
        .agg(
            n=("jid", "count"),
            min_num_atoms=("num_atoms", "min"),
            max_num_atoms=("num_atoms", "max"),
            mean_num_atoms=("num_atoms", "mean"),
            mean_abs_error_delta=("abs_error_delta", "mean"),
            win_rate=("candidate_wins", "mean"),
        )
        .reset_index()
    )
    atom_bins.to_csv(output_dir / "phase6_transfer_by_num_atoms_bin.csv", index=False)

    bandgap = (
        regime[bandgap_mask]
        .groupby(["comparison", "target_name", "bandgap_class"])
        .agg(
            n=("jid", "count"),
            mean_abs_error_delta=("abs_error_delta", "mean"),
            win_rate=("candidate_wins", "mean"),
        )
        .reset_index()
    )
    bandgap.to_csv(output_dir / "phase6_transfer_by_bandgap_class.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze Phase 6 per-sample transfer and error regimes."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--logs-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--targets", default=",".join(TARGETS))
    parser.add_argument("--fractions", default="0.10,0.25,0.50,1.0")
    parser.add_argument("--seeds", default="123,234,345")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    logs_dir = args.logs_dir or project_root / "results" / "logs"
    output_dir = args.output_dir or project_root / "results" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = [target.strip() for target in args.targets.split(",") if target.strip()]
    fractions = [float(value) for value in args.fractions.split(",") if value.strip()]
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]

    per_sample = _write_per_sample_transfer(
        project_root=project_root,
        logs_dir=logs_dir,
        output_dir=output_dir,
        targets=targets,
        fractions=fractions,
        seeds=seeds,
    )
    _write_regime_summaries(per_sample=per_sample, output_dir=output_dir)
    print(f"per_sample={output_dir / 'phase6_per_sample_transfer.csv'}")
    print(f"summary={output_dir / 'phase6_transfer_by_target_fraction.csv'}")
    print(f"target_bins={output_dir / 'phase6_transfer_by_target_value_bin.csv'}")
    print(f"atom_bins={output_dir / 'phase6_transfer_by_num_atoms_bin.csv'}")
    print(f"bandgap={output_dir / 'phase6_transfer_by_bandgap_class.csv'}")


if __name__ == "__main__":
    main()
