from __future__ import annotations

import argparse
from pathlib import Path

from alignn.data.jarvis import prepare_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="alignn-day2",
        description="Day 2 data preparation for the ALIGNN reimplementation project.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help="Download JARVIS data, inspect it, and generate train/val/test splits.",
    )
    prepare.add_argument("--dataset", default="dft_3d", help="JARVIS dataset name.")
    prepare.add_argument(
        "--target",
        default="formation_energy_peratom",
        help="Regression target column to keep.",
    )
    prepare.add_argument(
        "--max-samples",
        type=int,
        default=10000,
        help="Optional cap after filtering. Use 0 or a negative value for the full dataset.",
    )
    prepare.add_argument("--seed", type=int, default=42, help="Split seed.")
    prepare.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root containing data/, results/, and notebooks/ directories.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "prepare":
        prepare_dataset(
            project_root=args.project_root,
            dataset_name=args.dataset,
            target_column=args.target,
            max_samples=args.max_samples,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
