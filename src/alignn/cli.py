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

    import_splits = subparsers.add_parser(
        "import-splits",
        help="Create split CSVs from an ids_train_val_test.json file.",
    )
    import_splits.add_argument("--ids-json", type=Path, required=True)
    import_splits.add_argument("--dataset", default="dft_3d")
    import_splits.add_argument("--target", default="formation_energy_peratom")
    import_splits.add_argument("--project-root", type=Path, default=Path.cwd())

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

    alignn_forward = subparsers.add_parser(
        "alignn-forward",
        help="Run an ALIGNN forward pass on one prepared batch.",
    )
    alignn_forward.add_argument("--dataset", default="dft_3d")
    alignn_forward.add_argument("--target", default="formation_energy_peratom")
    alignn_forward.add_argument("--split", default="train")
    alignn_forward.add_argument("--batch-size", type=int, default=4)
    alignn_forward.add_argument("--hidden-dim", type=int, default=64)
    alignn_forward.add_argument("--alignn-layers", type=int, default=4)
    alignn_forward.add_argument("--gcn-layers", type=int, default=4)
    alignn_forward.add_argument("--cutoff", type=float, default=8.0)
    alignn_forward.add_argument("--max-neighbors", type=int, default=12)
    alignn_forward.add_argument("--device", default=None)
    alignn_forward.add_argument("--project-root", type=Path, default=Path.cwd())

    alignn_train = subparsers.add_parser(
        "alignn-train-small",
        help="Train ALIGNN on a small real subset and report validation metrics.",
    )
    alignn_train.add_argument("--dataset", default="dft_3d")
    alignn_train.add_argument("--target", default="formation_energy_peratom")
    alignn_train.add_argument("--train-split", default="train")
    alignn_train.add_argument("--val-split", default="val")
    alignn_train.add_argument("--test-split", default="test")
    alignn_train.add_argument("--train-fraction", type=float, default=1.0)
    alignn_train.add_argument("--train-subset-size", type=int, default=64)
    alignn_train.add_argument("--val-subset-size", type=int, default=16)
    alignn_train.add_argument("--test-subset-size", type=int, default=16)
    alignn_train.add_argument("--batch-size", type=int, default=4)
    alignn_train.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker processes. Use 0 for single-process loading.",
    )
    alignn_train.add_argument("--hidden-dim", type=int, default=64)
    alignn_train.add_argument("--alignn-layers", type=int, default=4)
    alignn_train.add_argument("--gcn-layers", type=int, default=4)
    alignn_train.add_argument("--cutoff", type=float, default=8.0)
    alignn_train.add_argument(
        "--cutoff-extra",
        type=float,
        default=0.0,
        help="Additional radius added to cutoff for neighbor search (mimics original ALIGNN cutoff_extra).",
    )
    alignn_train.add_argument("--max-neighbors", type=int, default=12)
    alignn_train.add_argument("--epochs", type=int, default=10)
    alignn_train.add_argument("--seed", type=int, default=123)
    alignn_train.add_argument("--learning-rate", type=float, default=1e-3)
    alignn_train.add_argument("--weight-decay", type=float, default=1e-5)
    alignn_train.add_argument("--loss", choices=["l1", "mse", "smoothl1"], default="l1")
    alignn_train.add_argument("--scheduler", choices=["onecycle", "none"], default="onecycle")
    alignn_train.add_argument(
        "--target-transform",
        choices=["none", "standardize", "log1p", "sqrt"],
        default="none",
        help="Transform targets during training while reporting metrics on the original scale.",
    )
    alignn_train.add_argument("--positive-weight", type=float, default=1.0)
    alignn_train.add_argument("--high-positive-weight", type=float, default=1.0)
    alignn_train.add_argument("--high-target-threshold", type=float, default=1.0)
    alignn_train.add_argument("--low-target-weight", type=float, default=1.0)
    alignn_train.add_argument("--low-target-threshold", type=float, default=0.0)
    alignn_train.add_argument("--mse-tail-weight", type=float, default=0.0)
    alignn_train.add_argument(
        "--readout",
        choices=["mean", "meanmax"],
        default="mean",
        help="Graph pooling used by ALIGNN before the regression head.",
    )
    alignn_train.add_argument(
        "--selection-metric",
        choices=["mae", "rmse"],
        default="mae",
        help="Validation metric used to select the checkpoint.",
    )
    alignn_train.add_argument(
        "--prediction-min",
        type=float,
        default=None,
        help="Optional lower bound applied to predictions for metrics and exported predictions.",
    )
    alignn_train.add_argument(
        "--energy-mult-natoms",
        action="store_true",
        help="Multiply model output by number of atoms inside the model forward pass.",
    )
    alignn_train.add_argument(
        "--penalty-factor",
        type=float,
        default=0.0,
        help="Short-bond penalty factor applied inside model forward pass.",
    )
    alignn_train.add_argument(
        "--penalty-threshold",
        type=float,
        default=1.0,
        help="Distance threshold for --penalty-factor.",
    )
    alignn_train.add_argument(
        "--use-amp",
        action="store_true",
        help="Use automatic mixed precision (float16) for faster training.",
    )
    alignn_train.add_argument(
        "--use-cudnn-benchmark",
        action="store_true",
        help="Enable cudnn.benchmark for faster convolutions on fixed-size inputs.",
    )
    alignn_train.add_argument(
        "--torch-compile",
        action="store_true",
        help="Use torch.compile for fused kernels (requires PyTorch 2.0+).",
    )
    alignn_train.add_argument(
        "--run-name",
        default="alignn_small_subset",
        help="Prefix for checkpoint, history, and prediction output files.",
    )
    alignn_train.add_argument(
        "--pretrained-multitask-checkpoint",
        type=Path,
        default=None,
        help="Optional multi-task checkpoint used to initialize the ALIGNN encoder.",
    )
    alignn_train.add_argument("--device", default=None)
    alignn_train.add_argument("--project-root", type=Path, default=Path.cwd())

    multitask_train = subparsers.add_parser(
        "alignn-train-multitask",
        help="Train shared-encoder ALIGNN with homogeneous multi-task batches.",
    )
    multitask_train.add_argument("--dataset", default="dft_3d")
    multitask_train.add_argument(
        "--targets",
        default=(
            "formation_energy_peratom,ehull,optb88vdw_bandgap,"
            "mbj_bandgap,bulk_modulus_kv"
        ),
        help="Comma-separated target columns with prepared split CSVs.",
    )
    multitask_train.add_argument("--train-split", default="train")
    multitask_train.add_argument("--val-split", default="val")
    multitask_train.add_argument("--test-split", default="test")
    multitask_train.add_argument("--train-fraction", type=float, default=1.0)
    multitask_train.add_argument("--train-subset-size", type=int, default=0)
    multitask_train.add_argument("--val-subset-size", type=int, default=0)
    multitask_train.add_argument("--test-subset-size", type=int, default=0)
    multitask_train.add_argument("--batch-size", type=int, default=16)
    multitask_train.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help=(
            "Approximate DataLoader worker budget. Multi-task training splits "
            "positive values across target loaders with at least one per target."
        ),
    )
    multitask_train.add_argument("--hidden-dim", type=int, default=64)
    multitask_train.add_argument("--head-hidden-dim", type=int, default=None)
    multitask_train.add_argument("--alignn-layers", type=int, default=4)
    multitask_train.add_argument("--gcn-layers", type=int, default=4)
    multitask_train.add_argument("--cutoff", type=float, default=8.0)
    multitask_train.add_argument("--max-neighbors", type=int, default=12)
    multitask_train.add_argument("--epochs", type=int, default=10)
    multitask_train.add_argument("--seed", type=int, default=123)
    multitask_train.add_argument("--learning-rate", type=float, default=1e-3)
    multitask_train.add_argument("--weight-decay", type=float, default=1e-5)
    multitask_train.add_argument("--loss", choices=["l1", "mse", "smoothl1"], default="smoothl1")
    multitask_train.add_argument("--scheduler", choices=["onecycle", "none"], default="onecycle")
    multitask_train.add_argument(
        "--readout",
        choices=["mean", "meanmax"],
        default="mean",
        help="Graph pooling used by the shared encoder.",
    )
    multitask_train.add_argument("--run-name", default="multitask_alignn")
    multitask_train.add_argument("--device", default=None)
    multitask_train.add_argument("--project-root", type=Path, default=Path.cwd())

    multitask_overfit = subparsers.add_parser(
        "alignn-overfit-multitask",
        help="Overfit a tiny homogeneous multi-task ALIGNN subset.",
    )
    multitask_overfit.add_argument("--dataset", default="dft_3d")
    multitask_overfit.add_argument(
        "--targets",
        default="formation_energy_peratom,ehull",
        help="Comma-separated target columns with prepared split CSVs.",
    )
    multitask_overfit.add_argument("--subset-size", type=int, default=8)
    multitask_overfit.add_argument("--batch-size", type=int, default=2)
    multitask_overfit.add_argument("--num-workers", type=int, default=4)
    multitask_overfit.add_argument("--hidden-dim", type=int, default=64)
    multitask_overfit.add_argument("--head-hidden-dim", type=int, default=None)
    multitask_overfit.add_argument("--alignn-layers", type=int, default=4)
    multitask_overfit.add_argument("--gcn-layers", type=int, default=4)
    multitask_overfit.add_argument("--cutoff", type=float, default=8.0)
    multitask_overfit.add_argument("--max-neighbors", type=int, default=12)
    multitask_overfit.add_argument("--epochs", type=int, default=50)
    multitask_overfit.add_argument("--seed", type=int, default=123)
    multitask_overfit.add_argument("--learning-rate", type=float, default=1e-3)
    multitask_overfit.add_argument("--weight-decay", type=float, default=1e-5)
    multitask_overfit.add_argument("--loss", choices=["l1", "mse", "smoothl1"], default="smoothl1")
    multitask_overfit.add_argument("--scheduler", choices=["onecycle", "none"], default="none")
    multitask_overfit.add_argument("--readout", choices=["mean", "meanmax"], default="mean")
    multitask_overfit.add_argument("--run-name", default="multitask_tiny_overfit")
    multitask_overfit.add_argument("--device", default=None)
    multitask_overfit.add_argument("--project-root", type=Path, default=Path.cwd())
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
    elif args.command == "import-splits":
        from alignn.data.jarvis import import_splits_from_ids_json

        import_splits_from_ids_json(
            project_root=args.project_root,
            ids_json=args.ids_json,
            dataset_name=args.dataset,
            target_column=args.target,
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
            seed=args.seed,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            loss_name=args.loss,
            scheduler_name=args.scheduler,
            target_transform=args.target_transform,
            positive_weight=args.positive_weight,
            high_positive_weight=args.high_positive_weight,
            high_target_threshold=args.high_target_threshold,
            low_target_weight=args.low_target_weight,
            low_target_threshold=args.low_target_threshold,
            mse_tail_weight=args.mse_tail_weight,
            prediction_min=args.prediction_min,
            selection_metric=args.selection_metric,
            readout=args.readout,
            energy_mult_natoms=args.energy_mult_natoms,
            penalty_factor=args.penalty_factor,
            penalty_threshold=args.penalty_threshold,
            use_amp=args.use_amp,
            use_cudnn_benchmark=args.use_cudnn_benchmark,
            torch_compile=args.torch_compile,
            pretrained_multitask_checkpoint=args.pretrained_multitask_checkpoint,
            run_name=args.run_name,
            device=args.device,
        )
    elif args.command == "alignn-train-small":
        from alignn.train.trainer import train_alignn_small_subset

        train_alignn_small_subset(
            project_root=args.project_root,
            dataset_name=args.dataset,
            target_column=args.target,
            train_split=args.train_split,
            val_split=args.val_split,
            test_split=args.test_split,
            train_fraction=args.train_fraction,
            train_subset_size=args.train_subset_size,
            val_subset_size=args.val_subset_size,
            test_subset_size=args.test_subset_size,
            batch_size=args.batch_size,
            hidden_dim=args.hidden_dim,
            alignn_layers=args.alignn_layers,
            gcn_layers=args.gcn_layers,
            cutoff=args.cutoff,
            max_neighbors=args.max_neighbors,
            epochs=args.epochs,
            seed=args.seed,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            loss_name=args.loss,
            scheduler_name=args.scheduler,
            target_transform=args.target_transform,
            positive_weight=args.positive_weight,
            high_positive_weight=args.high_positive_weight,
            high_target_threshold=args.high_target_threshold,
            low_target_weight=args.low_target_weight,
            low_target_threshold=args.low_target_threshold,
            mse_tail_weight=args.mse_tail_weight,
            prediction_min=args.prediction_min,
            selection_metric=args.selection_metric,
            readout=args.readout,
            energy_mult_natoms=args.energy_mult_natoms,
            penalty_factor=args.penalty_factor,
            penalty_threshold=args.penalty_threshold,
            use_amp=args.use_amp,
            use_cudnn_benchmark=args.use_cudnn_benchmark,
            torch_compile=args.torch_compile,
            pretrained_multitask_checkpoint=args.pretrained_multitask_checkpoint,
            run_name=args.run_name,
            device=args.device,
        )
    elif args.command == "alignn-train-multitask":
        from alignn.train.trainer import train_multitask_alignn

        train_multitask_alignn(
            project_root=args.project_root,
            dataset_name=args.dataset,
            targets=args.targets.split(","),
            train_split=args.train_split,
            val_split=args.val_split,
            test_split=args.test_split,
            train_fraction=args.train_fraction,
            train_subset_size=args.train_subset_size,
            val_subset_size=args.val_subset_size,
            test_subset_size=args.test_subset_size,
            batch_size=args.batch_size,
            hidden_dim=args.hidden_dim,
            head_hidden_dim=args.head_hidden_dim,
            alignn_layers=args.alignn_layers,
            gcn_layers=args.gcn_layers,
            cutoff=args.cutoff,
            max_neighbors=args.max_neighbors,
            epochs=args.epochs,
            seed=args.seed,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            loss_name=args.loss,
            scheduler_name=args.scheduler,
            readout=args.readout,
            run_name=args.run_name,
            device=args.device,
        )
    elif args.command == "alignn-overfit-multitask":
        from alignn.train.trainer import overfit_multitask_tiny_subset

        overfit_multitask_tiny_subset(
            project_root=args.project_root,
            dataset_name=args.dataset,
            targets=args.targets.split(","),
            subset_size=args.subset_size,
            batch_size=args.batch_size,
            hidden_dim=args.hidden_dim,
            head_hidden_dim=args.head_hidden_dim,
            alignn_layers=args.alignn_layers,
            gcn_layers=args.gcn_layers,
            cutoff=args.cutoff,
            max_neighbors=args.max_neighbors,
            epochs=args.epochs,
            seed=args.seed,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            loss_name=args.loss,
            scheduler_name=args.scheduler,
            readout=args.readout,
            run_name=args.run_name,
            device=args.device,
        )


if __name__ == "__main__":
    main()
