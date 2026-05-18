from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from alignn.data.dataset import (
    JarvisGraphDataset,
    TargetLabeledGraphDataset,
    collate_graph_samples,
    collate_graph_samples_with_line_graph,
    collate_multitask_graph_samples_with_line_graph,
)
from alignn.models.alignn_model import ALIGNNModel, MultiTaskALIGNNModel
from alignn.models.baseline_gnn import BaselineGNN


def _device_from_name(device: str | None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _make_baseline_dataset(
    project_root: Path,
    split: str,
    dataset_name: str,
    target_column: str,
    cutoff: float,
    max_neighbors: int,
) -> JarvisGraphDataset:
    return JarvisGraphDataset(
        project_root=project_root,
        split=split,
        dataset_name=dataset_name,
        target_column=target_column,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        compute_line_graph=False,
    )


def _make_alignn_dataset(
    project_root: Path,
    split: str,
    dataset_name: str,
    target_column: str,
    cutoff: float,
    max_neighbors: int,
) -> JarvisGraphDataset:
    return JarvisGraphDataset(
        project_root=project_root,
        split=split,
        dataset_name=dataset_name,
        target_column=target_column,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        compute_line_graph=True,
    )


def _regression_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, float]:
    errors = predictions - targets
    abs_errors = torch.abs(errors)
    metrics = {
        "mae": float(torch.mean(torch.abs(errors)).item()),
        "rmse": float(torch.sqrt(torch.mean(errors**2)).item()),
    }
    for name, mask in {
        "nonnegative": targets >= 0,
        "high_positive": targets >= 1,
    }.items():
        if bool(mask.any()):
            subset_errors = errors[mask]
            metrics[f"{name}_mae"] = float(torch.mean(torch.abs(subset_errors)).item())
            metrics[f"{name}_rmse"] = float(
                torch.sqrt(torch.mean(subset_errors**2)).item()
            )
        else:
            metrics[f"{name}_mae"] = float("nan")
            metrics[f"{name}_rmse"] = float("nan")
    metrics["p95_abs_error"] = float(torch.quantile(abs_errors, 0.95).item())
    metrics["max_abs_error"] = float(torch.max(abs_errors).item())
    return metrics


def _sample_weights(
    targets: torch.Tensor,
    positive_weight: float,
    high_positive_weight: float,
    high_target_threshold: float,
    low_target_weight: float,
    low_target_threshold: float,
) -> torch.Tensor:
    weights = torch.ones_like(targets)
    weights = torch.where(targets > 0, torch.full_like(weights, positive_weight), weights)
    weights = torch.where(
        targets > high_target_threshold,
        torch.full_like(weights, high_positive_weight),
        weights,
    )
    weights = torch.where(
        targets <= low_target_threshold,
        torch.full_like(weights, low_target_weight),
        weights,
    )
    return weights


def _weighted_regression_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    weight_targets: torch.Tensor,
    loss_name: str,
    positive_weight: float,
    high_positive_weight: float,
    high_target_threshold: float,
    low_target_weight: float,
    low_target_threshold: float,
    mse_tail_weight: float,
) -> torch.Tensor:
    if loss_name == "mse":
        per_sample = (predictions - targets) ** 2
    elif loss_name == "smoothl1":
        per_sample = torch.nn.functional.smooth_l1_loss(
            predictions,
            targets,
            reduction="none",
        )
    elif loss_name == "l1":
        per_sample = torch.abs(predictions - targets)
    else:
        raise ValueError(f"Unsupported loss: {loss_name}")

    weights = _sample_weights(
        weight_targets,
        positive_weight,
        high_positive_weight,
        high_target_threshold,
        low_target_weight,
        low_target_threshold,
    )
    loss = torch.mean(per_sample * weights)
    if mse_tail_weight > 0:
        tail_mse = (predictions - targets) ** 2
        loss = loss + mse_tail_weight * torch.mean(tail_mse * weights)
    return loss


class _TargetTransform:
    def __init__(self, name: str, targets: torch.Tensor) -> None:
        self.name = name
        self.mean = float(targets.mean().item())
        self.std = float(targets.std(unbiased=False).clamp_min(1e-8).item())
        if name in {"log1p", "sqrt"} and bool((targets < 0).any()):
            raise ValueError(f"{name} target transform requires nonnegative targets.")
        if name not in {"none", "standardize", "log1p", "sqrt"}:
            raise ValueError(f"Unsupported target transform: {name}")

    def forward(self, targets: torch.Tensor) -> torch.Tensor:
        if self.name == "standardize":
            return (targets - self.mean) / self.std
        if self.name == "log1p":
            return torch.log1p(targets)
        if self.name == "sqrt":
            return torch.sqrt(targets.clamp_min(0))
        return targets

    def inverse(self, predictions: torch.Tensor) -> torch.Tensor:
        if self.name == "standardize":
            return predictions * self.std + self.mean
        if self.name == "log1p":
            return torch.expm1(predictions).clamp_min(0)
        if self.name == "sqrt":
            return torch.square(predictions).clamp_min(0)
        return predictions


def _collect_targets(dataset: Subset | JarvisGraphDataset | TargetLabeledGraphDataset) -> torch.Tensor:
    if isinstance(dataset, Subset):
        parent_targets = _collect_targets(dataset.dataset)
        return parent_targets[list(dataset.indices)]
    if isinstance(dataset, TargetLabeledGraphDataset):
        return _collect_targets(dataset.dataset)
    targets = dataset.frame["target"]
    return torch.tensor(targets.astype(float).tolist(), dtype=torch.float32)


def _subset_or_full(
    dataset: Subset | JarvisGraphDataset | TargetLabeledGraphDataset,
    size: int,
) -> Subset | JarvisGraphDataset | TargetLabeledGraphDataset:
    if size <= 0:
        return dataset
    return Subset(dataset, range(min(size, len(dataset))))


def _fraction_subset(
    dataset: JarvisGraphDataset | TargetLabeledGraphDataset,
    fraction: float,
    seed: int,
) -> Subset | JarvisGraphDataset | TargetLabeledGraphDataset:
    if fraction <= 0 or fraction >= 1:
        return dataset
    subset_size = max(1, int(round(len(dataset) * fraction)))
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[
        :subset_size
    ].tolist()
    return Subset(dataset, indices)


def _evaluate_alignn(
    model: ALIGNNModel,
    loader: DataLoader,
    device: torch.device,
    target_transform: _TargetTransform | None = None,
    prediction_min: float | None = None,
    return_predictions: bool = False,
) -> dict[str, float] | tuple[dict[str, float], list[dict[str, float | str]]]:
    model.eval()
    predictions: list[torch.Tensor] = []
    targets_all: list[torch.Tensor] = []
    rows: list[dict[str, float | str]] = []
    with torch.no_grad():
        for graph_batch, line_graph_batch, targets, jids in loader:
            graph_batch = graph_batch.to(device)
            line_graph_batch = line_graph_batch.to(device)
            targets = targets.to(device)
            batch_predictions = model(graph_batch, line_graph_batch)
            if target_transform is not None:
                batch_predictions = target_transform.inverse(batch_predictions)
            if prediction_min is not None:
                batch_predictions = batch_predictions.clamp_min(prediction_min)
            predictions.append(batch_predictions.cpu())
            targets_all.append(targets.cpu())
            if return_predictions:
                for jid, target, prediction in zip(
                    jids,
                    targets.detach().cpu().tolist(),
                    batch_predictions.detach().cpu().tolist(),
                ):
                    rows.append(
                        {
                            "jid": str(jid),
                            "target": float(target),
                            "prediction": float(prediction),
                            "abs_error": abs(float(prediction) - float(target)),
                        }
                    )
    metrics = _regression_metrics(torch.cat(predictions), torch.cat(targets_all))
    if return_predictions:
        return metrics, rows
    return metrics


def _evaluate_multitask_target(
    model: MultiTaskALIGNNModel,
    target_name: str,
    loader: DataLoader,
    device: torch.device,
    target_transform: _TargetTransform,
    return_predictions: bool = False,
) -> dict[str, float] | tuple[dict[str, float], list[dict[str, float | str]]]:
    model.eval()
    predictions: list[torch.Tensor] = []
    standardized_predictions: list[torch.Tensor] = []
    targets_all: list[torch.Tensor] = []
    rows: list[dict[str, float | str]] = []
    with torch.no_grad():
        for graph_batch, line_graph_batch, targets, jids, _, _ in loader:
            graph_batch = graph_batch.to(device)
            line_graph_batch = line_graph_batch.to(device)
            targets = targets.to(device)
            batch_standardized = model(graph_batch, line_graph_batch, target_name)
            batch_predictions = target_transform.inverse(batch_standardized)
            predictions.append(batch_predictions.cpu())
            standardized_predictions.append(batch_standardized.cpu())
            targets_all.append(targets.cpu())
            if return_predictions:
                for jid, target, prediction in zip(
                    jids,
                    targets.detach().cpu().tolist(),
                    batch_predictions.detach().cpu().tolist(),
                ):
                    rows.append(
                        {
                            "jid": str(jid),
                            "target_name": target_name,
                            "target": float(target),
                            "prediction": float(prediction),
                            "abs_error": abs(float(prediction) - float(target)),
                        }
                    )
    targets_cat = torch.cat(targets_all)
    metrics = _regression_metrics(torch.cat(predictions), targets_cat)
    standardized_targets = target_transform.forward(targets_cat)
    standardized_errors = torch.cat(standardized_predictions) - standardized_targets
    metrics["standardized_mae"] = float(torch.mean(torch.abs(standardized_errors)).item())
    metrics["standardized_rmse"] = float(
        torch.sqrt(torch.mean(standardized_errors**2)).item()
    )
    if return_predictions:
        return metrics, rows
    return metrics


def run_baseline_forward_pass(
    project_root: Path,
    dataset_name: str = "dft_3d",
    target_column: str = "formation_energy_peratom",
    split: str = "train",
    batch_size: int = 4,
    hidden_dim: int = 64,
    num_layers: int = 4,
    cutoff: float = 8.0,
    max_neighbors: int = 12,
    device: str | None = None,
) -> None:
    dataset = _make_baseline_dataset(
        project_root=project_root,
        split=split,
        dataset_name=dataset_name,
        target_column=target_column,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_graph_samples,
    )
    graph_batch, targets, jids = next(iter(loader))

    run_device = _device_from_name(device)
    model = BaselineGNN(hidden_dim=hidden_dim, num_layers=num_layers).to(run_device)
    graph_batch = graph_batch.to(run_device)
    targets = targets.to(run_device)

    with torch.no_grad():
        predictions = model(graph_batch)

    print(
        "Baseline forward pass succeeded: "
        f"batch_size={len(jids)}, "
        f"num_nodes={graph_batch.num_nodes()}, "
        f"num_edges={graph_batch.num_edges()}, "
        f"pred_shape={tuple(predictions.shape)}, "
        f"target_shape={tuple(targets.shape)}."
    )


def overfit_baseline_tiny_subset(
    project_root: Path,
    dataset_name: str = "dft_3d",
    target_column: str = "formation_energy_peratom",
    split: str = "train",
    subset_size: int = 16,
    batch_size: int = 4,
    hidden_dim: int = 64,
    num_layers: int = 4,
    cutoff: float = 8.0,
    max_neighbors: int = 12,
    epochs: int = 50,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    device: str | None = None,
) -> None:
    project_root = project_root.resolve()
    run_device = _device_from_name(device)

    dataset = _make_baseline_dataset(
        project_root=project_root,
        split=split,
        dataset_name=dataset_name,
        target_column=target_column,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
    )
    subset = Subset(dataset, range(min(subset_size, len(dataset))))
    loader = DataLoader(
        subset,
        batch_size=min(batch_size, len(subset)),
        shuffle=True,
        collate_fn=collate_graph_samples,
    )

    model = BaselineGNN(hidden_dim=hidden_dim, num_layers=num_layers).to(run_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    criterion = nn.MSELoss()

    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_mae = 0.0
        batch_count = 0

        for graph_batch, targets, _ in loader:
            graph_batch = graph_batch.to(run_device)
            targets = targets.to(run_device)

            predictions = model(graph_batch)
            loss = criterion(predictions, targets)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            epoch_mae += float(torch.mean(torch.abs(predictions - targets)).item())
            batch_count += 1

        mean_loss = epoch_loss / max(batch_count, 1)
        mean_mae = epoch_mae / max(batch_count, 1)
        history.append({"epoch": epoch, "loss": mean_loss, "mae": mean_mae})

        if mean_loss < best_loss:
            best_loss = mean_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        print(f"epoch={epoch:03d} loss={mean_loss:.6f} mae={mean_mae:.6f}")

    checkpoints_dir = project_root / "results" / "checkpoints"
    logs_dir = project_root / "results" / "logs"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoints_dir / "baseline_tiny_overfit.pt"
    torch.save(
        {
            "model_state_dict": best_state if best_state is not None else model.state_dict(),
            "dataset_name": dataset_name,
            "target_column": target_column,
            "split": split,
            "subset_size": len(subset),
            "best_loss": best_loss,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
        },
        checkpoint_path,
    )

    history_path = logs_dir / "baseline_tiny_overfit_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "loss", "mae"])
        writer.writeheader()
        writer.writerows(history)

    print(
        "Baseline tiny-subset overfit run finished: "
        f"best_loss={best_loss:.6f}, "
        f"checkpoint={checkpoint_path}, "
        f"history={history_path}."
    )


def run_alignn_forward_pass(
    project_root: Path,
    dataset_name: str = "dft_3d",
    target_column: str = "formation_energy_peratom",
    split: str = "train",
    batch_size: int = 4,
    hidden_dim: int = 64,
    alignn_layers: int = 4,
    gcn_layers: int = 4,
    cutoff: float = 8.0,
    max_neighbors: int = 12,
    device: str | None = None,
) -> None:
    dataset = _make_alignn_dataset(
        project_root=project_root,
        split=split,
        dataset_name=dataset_name,
        target_column=target_column,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_graph_samples_with_line_graph,
    )
    graph_batch, line_graph_batch, targets, jids = next(iter(loader))

    run_device = _device_from_name(device)
    model = ALIGNNModel(
        hidden_dim=hidden_dim,
        alignn_layers=alignn_layers,
        gcn_layers=gcn_layers,
    ).to(run_device)
    graph_batch = graph_batch.to(run_device)
    line_graph_batch = line_graph_batch.to(run_device)
    targets = targets.to(run_device)

    with torch.no_grad():
        predictions = model(graph_batch, line_graph_batch)

    print(
        "ALIGNN forward pass succeeded: "
        f"batch_size={len(jids)}, "
        f"num_nodes={graph_batch.num_nodes()}, "
        f"num_edges={graph_batch.num_edges()}, "
        f"line_graph_nodes={line_graph_batch.num_nodes()}, "
        f"line_graph_edges={line_graph_batch.num_edges()}, "
        f"pred_shape={tuple(predictions.shape)}, "
        f"target_shape={tuple(targets.shape)}."
    )


def train_alignn_small_subset(
    project_root: Path,
    dataset_name: str = "dft_3d",
    target_column: str = "formation_energy_peratom",
    train_split: str = "train",
    val_split: str = "val",
    test_split: str = "test",
    train_fraction: float = 1.0,
    train_subset_size: int = 64,
    val_subset_size: int = 16,
    test_subset_size: int = 16,
    batch_size: int = 4,
    hidden_dim: int = 64,
    alignn_layers: int = 4,
    gcn_layers: int = 4,
    cutoff: float = 8.0,
    max_neighbors: int = 12,
    epochs: int = 10,
    seed: int = 123,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    loss_name: str = "l1",
    scheduler_name: str = "onecycle",
    target_transform: str = "none",
    positive_weight: float = 1.0,
    high_positive_weight: float = 1.0,
    high_target_threshold: float = 1.0,
    low_target_weight: float = 1.0,
    low_target_threshold: float = 0.0,
    mse_tail_weight: float = 0.0,
    prediction_min: float | None = None,
    selection_metric: str = "mae",
    readout: str = "mean",
    pretrained_multitask_checkpoint: Path | None = None,
    run_name: str = "alignn_small_subset",
    device: str | None = None,
) -> None:
    project_root = project_root.resolve()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    run_device = _device_from_name(device)

    train_dataset = _make_alignn_dataset(
        project_root=project_root,
        split=train_split,
        dataset_name=dataset_name,
        target_column=target_column,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
    )
    val_dataset = _make_alignn_dataset(
        project_root=project_root,
        split=val_split,
        dataset_name=dataset_name,
        target_column=target_column,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
    )
    test_dataset = _make_alignn_dataset(
        project_root=project_root,
        split=test_split,
        dataset_name=dataset_name,
        target_column=target_column,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
    )
    train_subset = _subset_or_full(
        _fraction_subset(train_dataset, train_fraction, seed),
        train_subset_size,
    )
    val_subset = _subset_or_full(val_dataset, val_subset_size)
    test_subset = _subset_or_full(test_dataset, test_subset_size)
    transform = _TargetTransform(
        name=target_transform,
        targets=_collect_targets(train_subset),
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=min(batch_size, len(train_subset)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=collate_graph_samples_with_line_graph,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=min(batch_size, len(val_subset)),
        shuffle=False,
        drop_last=False,
        collate_fn=collate_graph_samples_with_line_graph,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=min(batch_size, len(test_subset)),
        shuffle=False,
        drop_last=False,
        collate_fn=collate_graph_samples_with_line_graph,
    )

    model = ALIGNNModel(
        hidden_dim=hidden_dim,
        alignn_layers=alignn_layers,
        gcn_layers=gcn_layers,
        readout=readout,
    ).to(run_device)
    if pretrained_multitask_checkpoint is not None:
        checkpoint = torch.load(pretrained_multitask_checkpoint, map_location="cpu")
        if "encoder_state_dict" in checkpoint:
            checkpoint_state = {
                f"encoder.{key}": value
                for key, value in checkpoint["encoder_state_dict"].items()
            }
        else:
            checkpoint_state = checkpoint.get("model_state_dict", checkpoint)
        model_state = model.state_dict()
        encoder_state = {
            key: value
            for key, value in checkpoint_state.items()
            if key.startswith("encoder.")
            and key in model_state
            and tuple(model_state[key].shape) == tuple(value.shape)
        }
        if not encoder_state:
            raise ValueError(
                "No compatible encoder weights found in "
                f"{pretrained_multitask_checkpoint}."
            )
        model_state.update(encoder_state)
        model.load_state_dict(model_state)
        print(
            "Loaded multi-task encoder weights: "
            f"{len(encoder_state)} tensors from {pretrained_multitask_checkpoint}."
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if loss_name not in {"mse", "smoothl1", "l1"}:
        raise ValueError(f"Unsupported loss: {loss_name}")
    scheduler = None
    if scheduler_name == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            epochs=epochs,
            steps_per_epoch=max(len(train_loader), 1),
            pct_start=0.3,
        )
    elif scheduler_name == "none":
        scheduler = None
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    history: list[dict[str, float]] = []
    if selection_metric not in {"mae", "rmse"}:
        raise ValueError(f"Unsupported selection metric: {selection_metric}")
    best_val_score = float("inf")
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: list[float] = []
        train_predictions: list[torch.Tensor] = []
        train_targets: list[torch.Tensor] = []

        for graph_batch, line_graph_batch, targets, _ in train_loader:
            graph_batch = graph_batch.to(run_device)
            line_graph_batch = line_graph_batch.to(run_device)
            targets = targets.to(run_device)

            predictions = model(graph_batch, line_graph_batch)
            loss_targets = transform.forward(targets)
            loss = _weighted_regression_loss(
                predictions=predictions,
                targets=loss_targets,
                weight_targets=targets,
                loss_name=loss_name,
                positive_weight=positive_weight,
                high_positive_weight=high_positive_weight,
                high_target_threshold=high_target_threshold,
                low_target_weight=low_target_weight,
                low_target_threshold=low_target_threshold,
                mse_tail_weight=mse_tail_weight,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            train_losses.append(float(loss.item()))
            metric_predictions = transform.inverse(predictions)
            if prediction_min is not None:
                metric_predictions = metric_predictions.clamp_min(prediction_min)
            train_predictions.append(metric_predictions.detach().cpu())
            train_targets.append(targets.detach().cpu())

        train_metrics = _regression_metrics(
            torch.cat(train_predictions),
            torch.cat(train_targets),
        )
        val_metrics = _evaluate_alignn(
            model,
            val_loader,
            run_device,
            target_transform=transform,
            prediction_min=prediction_min,
        )
        mean_loss = sum(train_losses) / max(len(train_losses), 1)
        history.append(
            {
                "epoch": epoch,
                "loss": mean_loss,
                "train_mae": train_metrics["mae"],
                "train_rmse": train_metrics["rmse"],
                "val_mae": val_metrics["mae"],
                "val_rmse": val_metrics["rmse"],
                "val_nonnegative_mae": val_metrics["nonnegative_mae"],
                "val_high_positive_mae": val_metrics["high_positive_mae"],
                "val_p95_abs_error": val_metrics["p95_abs_error"],
                "val_max_abs_error": val_metrics["max_abs_error"],
            }
        )

        val_score = val_metrics[selection_metric]
        if val_score < best_val_score:
            best_val_score = val_score
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        print(
            f"epoch={epoch:03d} loss={mean_loss:.6f} "
            f"train_mae={train_metrics['mae']:.6f} val_mae={val_metrics['mae']:.6f} "
            f"val_rmse={val_metrics['rmse']:.6f} "
            f"val_high_pos_mae={val_metrics['high_positive_mae']:.6f} "
            f"val_p95={val_metrics['p95_abs_error']:.6f}"
        )

    checkpoints_dir = project_root / "results" / "checkpoints"
    logs_dir = project_root / "results" / "logs"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    safe_run_name = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in run_name
    ).strip("_") or "alignn_small_subset"
    checkpoint_path = checkpoints_dir / f"{safe_run_name}.pt"
    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics, test_rows = _evaluate_alignn(
        model,
        test_loader,
        run_device,
        target_transform=transform,
        prediction_min=prediction_min,
        return_predictions=True,
    )
    torch.save(
        {
            "model_state_dict": best_state if best_state is not None else model.state_dict(),
            "dataset_name": dataset_name,
            "target_column": target_column,
            "train_subset_size": len(train_subset),
            "train_fraction": train_fraction,
            "val_subset_size": len(val_subset),
            "test_subset_size": len(test_subset),
            "seed": seed,
            "selection_metric": selection_metric,
            "best_val_score": best_val_score,
            "test_mae": test_metrics["mae"],
            "test_rmse": test_metrics["rmse"],
            "hidden_dim": hidden_dim,
            "alignn_layers": alignn_layers,
            "gcn_layers": gcn_layers,
            "loss": loss_name,
            "scheduler": scheduler_name,
            "target_transform": target_transform,
            "readout": readout,
            "pretrained_multitask_checkpoint": str(pretrained_multitask_checkpoint)
            if pretrained_multitask_checkpoint is not None
            else None,
            "positive_weight": positive_weight,
            "high_positive_weight": high_positive_weight,
            "high_target_threshold": high_target_threshold,
            "low_target_weight": low_target_weight,
            "low_target_threshold": low_target_threshold,
            "mse_tail_weight": mse_tail_weight,
            "prediction_min": prediction_min,
            "test_nonnegative_mae": test_metrics["nonnegative_mae"],
            "test_high_positive_mae": test_metrics["high_positive_mae"],
            "test_p95_abs_error": test_metrics["p95_abs_error"],
            "test_max_abs_error": test_metrics["max_abs_error"],
        },
        checkpoint_path,
    )

    history_path = logs_dir / f"{safe_run_name}_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "loss",
                "train_mae",
                "train_rmse",
                "val_mae",
                "val_rmse",
                "val_nonnegative_mae",
                "val_high_positive_mae",
                "val_p95_abs_error",
                "val_max_abs_error",
            ],
        )
        writer.writeheader()
        writer.writerows(history)

    predictions_path = logs_dir / f"{safe_run_name}_test_predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["jid", "target", "prediction", "abs_error"],
        )
        writer.writeheader()
        writer.writerows(test_rows)

    metrics_path = logs_dir / f"{safe_run_name}_test_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "target": target_column,
                "dataset_name": dataset_name,
                "seed": seed,
                "train_fraction": train_fraction,
                "sizes": {
                    "train": len(train_subset),
                    "val": len(val_subset),
                    "test": len(test_subset),
                },
                "config": {
                    "hidden_dim": hidden_dim,
                    "alignn_layers": alignn_layers,
                    "gcn_layers": gcn_layers,
                    "loss": loss_name,
                    "scheduler": scheduler_name,
                    "target_transform": target_transform,
                    "readout": readout,
                    "pretrained_multitask_checkpoint": str(pretrained_multitask_checkpoint)
                    if pretrained_multitask_checkpoint is not None
                    else None,
                },
                "best_val_score": best_val_score,
                "selection_metric": selection_metric,
                "test_metrics": test_metrics,
                "checkpoint": str(checkpoint_path),
                "history": str(history_path),
                "predictions": str(predictions_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "ALIGNN small-subset training finished: "
        f"best_val_{selection_metric}={best_val_score:.6f}, "
        f"test_mae={test_metrics['mae']:.6f}, "
        f"test_rmse={test_metrics['rmse']:.6f}, "
        f"test_high_pos_mae={test_metrics['high_positive_mae']:.6f}, "
        f"test_p95={test_metrics['p95_abs_error']:.6f}, "
        f"checkpoint={checkpoint_path}, "
        f"history={history_path}, "
        f"predictions={predictions_path}, "
        f"metrics={metrics_path}."
    )


def train_multitask_alignn(
    project_root: Path,
    targets: list[str],
    dataset_name: str = "dft_3d",
    train_split: str = "train",
    val_split: str = "val",
    test_split: str = "test",
    train_fraction: float = 1.0,
    train_subset_size: int = 0,
    val_subset_size: int = 0,
    test_subset_size: int = 0,
    batch_size: int = 16,
    hidden_dim: int = 64,
    head_hidden_dim: int | None = None,
    alignn_layers: int = 4,
    gcn_layers: int = 4,
    cutoff: float = 8.0,
    max_neighbors: int = 12,
    epochs: int = 10,
    seed: int = 123,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    loss_name: str = "smoothl1",
    scheduler_name: str = "onecycle",
    readout: str = "mean",
    run_name: str = "multitask_alignn",
    device: str | None = None,
) -> None:
    """Train a shared ALIGNN encoder with homogeneous target batches."""

    project_root = project_root.resolve()
    target_names = [target.strip() for target in targets if target.strip()]
    if not target_names:
        raise ValueError("At least one target is required for multi-task training.")
    if loss_name not in {"mse", "smoothl1", "l1"}:
        raise ValueError(f"Unsupported loss: {loss_name}")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    run_device = _device_from_name(device)

    train_loaders: dict[str, DataLoader] = {}
    val_loaders: dict[str, DataLoader] = {}
    test_loaders: dict[str, DataLoader] = {}
    transforms: dict[str, _TargetTransform] = {}
    sizes: dict[str, dict[str, int]] = {}

    for target_index, target_name in enumerate(target_names):
        train_dataset = TargetLabeledGraphDataset(
            _make_alignn_dataset(
                project_root=project_root,
                split=train_split,
                dataset_name=dataset_name,
                target_column=target_name,
                cutoff=cutoff,
                max_neighbors=max_neighbors,
            ),
            target_name=target_name,
            target_id=target_index,
        )
        val_dataset = TargetLabeledGraphDataset(
            _make_alignn_dataset(
                project_root=project_root,
                split=val_split,
                dataset_name=dataset_name,
                target_column=target_name,
                cutoff=cutoff,
                max_neighbors=max_neighbors,
            ),
            target_name=target_name,
            target_id=target_index,
        )
        test_dataset = TargetLabeledGraphDataset(
            _make_alignn_dataset(
                project_root=project_root,
                split=test_split,
                dataset_name=dataset_name,
                target_column=target_name,
                cutoff=cutoff,
                max_neighbors=max_neighbors,
            ),
            target_name=target_name,
            target_id=target_index,
        )
        train_subset = _subset_or_full(
            _fraction_subset(train_dataset, train_fraction, seed + target_index),
            train_subset_size,
        )
        val_subset = _subset_or_full(val_dataset, val_subset_size)
        test_subset = _subset_or_full(test_dataset, test_subset_size)
        transforms[target_name] = _TargetTransform(
            name="standardize",
            targets=_collect_targets(train_subset),
        )
        sizes[target_name] = {
            "train": len(train_subset),
            "val": len(val_subset),
            "test": len(test_subset),
        }
        train_loaders[target_name] = DataLoader(
            train_subset,
            batch_size=min(batch_size, len(train_subset)),
            shuffle=True,
            generator=torch.Generator().manual_seed(seed + target_index),
            collate_fn=collate_multitask_graph_samples_with_line_graph,
        )
        val_loaders[target_name] = DataLoader(
            val_subset,
            batch_size=min(batch_size, len(val_subset)),
            shuffle=False,
            drop_last=False,
            collate_fn=collate_multitask_graph_samples_with_line_graph,
        )
        test_loaders[target_name] = DataLoader(
            test_subset,
            batch_size=min(batch_size, len(test_subset)),
            shuffle=False,
            drop_last=False,
            collate_fn=collate_multitask_graph_samples_with_line_graph,
        )

    model = MultiTaskALIGNNModel(
        target_names=target_names,
        hidden_dim=hidden_dim,
        alignn_layers=alignn_layers,
        gcn_layers=gcn_layers,
        readout=readout,
        head_hidden_dim=head_hidden_dim,
    ).to(run_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    max_steps_per_epoch = max(len(loader) for loader in train_loaders.values())
    scheduler = None
    if scheduler_name == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            epochs=epochs,
            steps_per_epoch=max_steps_per_epoch * len(target_names),
            pct_start=0.3,
        )
    elif scheduler_name != "none":
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    history: list[dict[str, float | int | str]] = []
    best_val_score = float("inf")
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, epochs + 1):
        model.train()
        iterators = {target: iter(loader) for target, loader in train_loaders.items()}
        train_loss_by_target: dict[str, list[float]] = {target: [] for target in target_names}
        train_predictions: dict[str, list[torch.Tensor]] = {target: [] for target in target_names}
        train_targets: dict[str, list[torch.Tensor]] = {target: [] for target in target_names}

        for _ in range(max_steps_per_epoch):
            for target_name in target_names:
                try:
                    graph_batch, line_graph_batch, targets_batch, _, _, _ = next(
                        iterators[target_name]
                    )
                except StopIteration:
                    iterators[target_name] = iter(train_loaders[target_name])
                    graph_batch, line_graph_batch, targets_batch, _, _, _ = next(
                        iterators[target_name]
                    )

                graph_batch = graph_batch.to(run_device)
                line_graph_batch = line_graph_batch.to(run_device)
                targets_batch = targets_batch.to(run_device)
                predictions = model(graph_batch, line_graph_batch, target_name)
                loss_targets = transforms[target_name].forward(targets_batch)
                if loss_name == "mse":
                    loss = torch.nn.functional.mse_loss(predictions, loss_targets)
                elif loss_name == "smoothl1":
                    loss = torch.nn.functional.smooth_l1_loss(predictions, loss_targets)
                else:
                    loss = torch.nn.functional.l1_loss(predictions, loss_targets)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                train_loss_by_target[target_name].append(float(loss.item()))
                train_predictions[target_name].append(
                    transforms[target_name].inverse(predictions).detach().cpu()
                )
                train_targets[target_name].append(targets_batch.detach().cpu())

        val_metrics_by_target: dict[str, dict[str, float]] = {}
        val_scores: list[float] = []
        for target_name in target_names:
            train_metrics = _regression_metrics(
                torch.cat(train_predictions[target_name]),
                torch.cat(train_targets[target_name]),
            )
            val_metrics = _evaluate_multitask_target(
                model=model,
                target_name=target_name,
                loader=val_loaders[target_name],
                device=run_device,
                target_transform=transforms[target_name],
            )
            val_metrics_by_target[target_name] = val_metrics
            val_scores.append(val_metrics["standardized_mae"])
            history.append(
                {
                    "epoch": epoch,
                    "target_name": target_name,
                    "train_loss": sum(train_loss_by_target[target_name])
                    / max(len(train_loss_by_target[target_name]), 1),
                    "train_mae": train_metrics["mae"],
                    "train_rmse": train_metrics["rmse"],
                    "val_mae": val_metrics["mae"],
                    "val_rmse": val_metrics["rmse"],
                    "val_p95_abs_error": val_metrics["p95_abs_error"],
                    "val_standardized_mae": val_metrics["standardized_mae"],
                }
            )

        val_score = sum(val_scores) / max(len(val_scores), 1)
        if val_score < best_val_score:
            best_val_score = val_score
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        val_summary = " ".join(
            f"{target}:val_mae={val_metrics_by_target[target]['mae']:.6f}"
            for target in target_names
        )
        print(
            f"epoch={epoch:03d} avg_val_standardized_mae={val_score:.6f} "
            f"{val_summary}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoints_dir = project_root / "results" / "checkpoints"
    logs_dir = project_root / "results" / "logs"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    safe_run_name = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in run_name
    ).strip("_") or "multitask_alignn"

    test_metrics_by_target: dict[str, dict[str, float]] = {}
    all_test_rows: list[dict[str, float | str]] = []
    for target_name in target_names:
        test_metrics, test_rows = _evaluate_multitask_target(
            model=model,
            target_name=target_name,
            loader=test_loaders[target_name],
            device=run_device,
            target_transform=transforms[target_name],
            return_predictions=True,
        )
        test_metrics_by_target[target_name] = test_metrics
        all_test_rows.extend(test_rows)

    best_epoch_by_target: dict[str, int] = {}
    for target_name in target_names:
        target_history = [row for row in history if row["target_name"] == target_name]
        if target_history:
            best_row = min(
                target_history,
                key=lambda row: float(row["val_standardized_mae"]),
            )
            best_epoch_by_target[target_name] = int(best_row["epoch"])

    checkpoint_path = checkpoints_dir / f"{safe_run_name}.pt"
    encoder_checkpoint_path = checkpoints_dir / f"{safe_run_name}_encoder.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "target_names": target_names,
            "dataset_name": dataset_name,
            "sizes": sizes,
            "seed": seed,
            "best_val_standardized_mae": best_val_score,
            "hidden_dim": hidden_dim,
            "head_hidden_dim": head_hidden_dim,
            "alignn_layers": alignn_layers,
            "gcn_layers": gcn_layers,
            "loss": loss_name,
            "scheduler": scheduler_name,
            "readout": readout,
            "target_transforms": {
                target: {"mean": transforms[target].mean, "std": transforms[target].std}
                for target in target_names
            },
            "best_epoch_by_target": best_epoch_by_target,
            "test_metrics": test_metrics_by_target,
        },
        checkpoint_path,
    )
    torch.save(
        {
            "encoder_state_dict": model.encoder.state_dict(),
            "target_names": target_names,
            "dataset_name": dataset_name,
            "hidden_dim": hidden_dim,
            "alignn_layers": alignn_layers,
            "gcn_layers": gcn_layers,
            "readout": readout,
            "target_transforms": {
                target: {"mean": transforms[target].mean, "std": transforms[target].std}
                for target in target_names
            },
        },
        encoder_checkpoint_path,
    )

    history_path = logs_dir / f"{safe_run_name}_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "target_name",
                "train_loss",
                "train_mae",
                "train_rmse",
                "val_mae",
                "val_rmse",
                "val_p95_abs_error",
                "val_standardized_mae",
            ],
        )
        writer.writeheader()
        writer.writerows(history)

    predictions_path = logs_dir / f"{safe_run_name}_test_predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["jid", "target_name", "target", "prediction", "abs_error"],
        )
        writer.writeheader()
        writer.writerows(all_test_rows)

    metrics_path = logs_dir / f"{safe_run_name}_test_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "best_val_standardized_mae": best_val_score,
                "targets": target_names,
                "sizes": sizes,
                "best_epoch_by_target": best_epoch_by_target,
                "test_metrics": test_metrics_by_target,
                "checkpoint": str(checkpoint_path),
                "encoder_checkpoint": str(encoder_checkpoint_path),
                "history": str(history_path),
                "predictions": str(predictions_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    metric_summary = " ".join(
        f"{target}:test_mae={test_metrics_by_target[target]['mae']:.6f}"
        for target in target_names
    )
    print(
        "Multi-task ALIGNN training finished: "
        f"best_val_standardized_mae={best_val_score:.6f}, "
        f"{metric_summary}, "
        f"checkpoint={checkpoint_path}, "
        f"encoder_checkpoint={encoder_checkpoint_path}, "
        f"history={history_path}, "
        f"predictions={predictions_path}, "
        f"metrics={metrics_path}."
    )


def overfit_multitask_tiny_subset(
    project_root: Path,
    targets: list[str],
    dataset_name: str = "dft_3d",
    subset_size: int = 8,
    batch_size: int = 2,
    hidden_dim: int = 64,
    head_hidden_dim: int | None = None,
    alignn_layers: int = 4,
    gcn_layers: int = 4,
    cutoff: float = 8.0,
    max_neighbors: int = 12,
    epochs: int = 50,
    seed: int = 123,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    loss_name: str = "smoothl1",
    scheduler_name: str = "none",
    readout: str = "mean",
    run_name: str = "multitask_tiny_overfit",
    device: str | None = None,
) -> None:
    """Tiny multi-task overfit path for the plan's Phase 2 smoke criterion."""

    train_multitask_alignn(
        project_root=project_root,
        targets=targets,
        dataset_name=dataset_name,
        train_subset_size=subset_size,
        val_subset_size=subset_size,
        test_subset_size=subset_size,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        head_hidden_dim=head_hidden_dim,
        alignn_layers=alignn_layers,
        gcn_layers=gcn_layers,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        epochs=epochs,
        seed=seed,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        loss_name=loss_name,
        scheduler_name=scheduler_name,
        readout=readout,
        run_name=run_name,
        device=device,
    )
