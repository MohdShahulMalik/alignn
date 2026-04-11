from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from jarvis.db.figshare import data as jarvis_data

from alignn.data.splits import create_split_frames


def _ensure_dirs(project_root: Path) -> dict[str, Path]:
    paths = {
        "raw": project_root / "data" / "raw",
        "processed": project_root / "data" / "processed",
        "splits": project_root / "data" / "splits",
        "tables": project_root / "results" / "tables",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _load_records(dataset_name: str, raw_dir: Path) -> list[dict]:
    records = jarvis_data(dataset=dataset_name, store_dir=str(raw_dir))
    if not records:
        raise RuntimeError(f"Dataset '{dataset_name}' returned no records.")
    return records


def _build_dataframe(records: list[dict], target_column: str) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    required = ["jid", "atoms", target_column]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(
            "Missing required columns in JARVIS data: " + ", ".join(sorted(missing))
        )

    filtered = frame.loc[frame[target_column].notna() & frame["atoms"].notna()].copy()
    filtered["target"] = filtered[target_column].astype(float)
    filtered["num_atoms"] = filtered["atoms"].map(
        lambda atoms: len(atoms["elements"]) if isinstance(atoms, dict) else None
    )
    return filtered


def _write_summary(
    frame: pd.DataFrame,
    dataset_name: str,
    target_column: str,
    max_samples: int,
    output_path: Path,
) -> None:
    summary = {
        "dataset": dataset_name,
        "target": target_column,
        "rows_after_filter": int(len(frame)),
        "used_max_samples": int(max_samples) if max_samples and max_samples > 0 else None,
        "target_describe": frame["target"].describe().to_dict(),
        "num_atoms_describe": frame["num_atoms"].dropna().describe().to_dict(),
        "example_jids": frame["jid"].head(10).tolist(),
    }
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def prepare_dataset(
    project_root: Path,
    dataset_name: str,
    target_column: str,
    max_samples: int,
    seed: int,
) -> None:
    project_root = project_root.resolve()
    paths = _ensure_dirs(project_root)

    records = _load_records(dataset_name=dataset_name, raw_dir=paths["raw"])
    frame = _build_dataframe(records=records, target_column=target_column)

    if max_samples and max_samples > 0:
        frame = frame.sample(n=min(max_samples, len(frame)), random_state=seed).copy()

    summary_frame = frame[["jid", target_column, "target", "num_atoms"]].copy()
    summary_frame.to_csv(
        paths["processed"] / f"{dataset_name}_{target_column}_summary.csv",
        index=False,
    )

    split_frames = create_split_frames(frame=summary_frame, seed=seed)
    for split_name, split_frame in split_frames.items():
        split_frame.to_csv(
            paths["splits"] / f"{dataset_name}_{target_column}_{split_name}.csv",
            index=False,
        )

    _write_summary(
        frame=summary_frame,
        dataset_name=dataset_name,
        target_column=target_column,
        max_samples=max_samples,
        output_path=paths["tables"] / f"{dataset_name}_{target_column}_inspection.json",
    )

    print(
        f"Prepared {dataset_name} for target {target_column}: "
        f"{len(summary_frame)} rows, "
        f"{len(split_frames['train'])} train / "
        f"{len(split_frames['val'])} val / "
        f"{len(split_frames['test'])} test."
    )
