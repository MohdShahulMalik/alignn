from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="alignn",
        description="ALIGNN reimplementation CLI.",
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

    baseline_forward = subparsers.add_parser(
        "baseline-forward",
        help="Run a baseline GNN forward pass on one prepared batch.",
    )
    baseline_forward.add_argument("--dataset", default="dft_3d")
    baseline_forward.add_argument("--target", default="formation_energy_peratom")
    baseline_forward.add_argument("--split", default="train")
    baseline_forward.add_argument("--batch-size", type=int, default=4)
    baseline_forward.add_argument("--hidden-dim", type=int, default=64)
    baseline_forward.add_argument("--num-layers", type=int, default=4)
    baseline_forward.add_argument("--cutoff", type=float, default=8.0)
    baseline_forward.add_argument("--max-neighbors", type=int, default=12)
    baseline_forward.add_argument("--device", default=None)
    baseline_forward.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
    )

    baseline_overfit = subparsers.add_parser(
        "baseline-overfit",
        help="Train the baseline GNN on a tiny subset to confirm it can overfit.",
    )
    baseline_overfit.add_argument("--dataset", default="dft_3d")
    baseline_overfit.add_argument("--target", default="formation_energy_peratom")
    baseline_overfit.add_argument("--split", default="train")
    baseline_overfit.add_argument("--subset-size", type=int, default=16)
    baseline_overfit.add_argument("--batch-size", type=int, default=4)
    baseline_overfit.add_argument("--hidden-dim", type=int, default=64)
    baseline_overfit.add_argument("--num-layers", type=int, default=4)
    baseline_overfit.add_argument("--cutoff", type=float, default=8.0)
    baseline_overfit.add_argument("--max-neighbors", type=int, default=12)
    baseline_overfit.add_argument("--epochs", type=int, default=50)
    baseline_overfit.add_argument("--learning-rate", type=float, default=1e-3)
    baseline_overfit.add_argument("--weight-decay", type=float, default=1e-5)
    baseline_overfit.add_argument("--device", default=None)
    baseline_overfit.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "prepare":
        from alignn.data.jarvis import prepare_dataset

        prepare_dataset(
            project_root=args.project_root,
            dataset_name=args.dataset,
            target_column=args.target,
            max_samples=args.max_samples,
            seed=args.seed,
        )
    elif args.command == "baseline-forward":
        from alignn.train.trainer import run_baseline_forward_pass

        run_baseline_forward_pass(
            project_root=args.project_root,
            dataset_name=args.dataset,
            target_column=args.target,
            split=args.split,
            batch_size=args.batch_size,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            cutoff=args.cutoff,
            max_neighbors=args.max_neighbors,
            device=args.device,
        )
    elif args.command == "baseline-overfit":
        from alignn.train.trainer import overfit_baseline_tiny_subset

        overfit_baseline_tiny_subset(
            project_root=args.project_root,
            dataset_name=args.dataset,
            target_column=args.target,
            split=args.split,
            subset_size=args.subset_size,
            batch_size=args.batch_size,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            cutoff=args.cutoff,
            max_neighbors=args.max_neighbors,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=args.device,
        )


if __name__ == "__main__":
    main()
