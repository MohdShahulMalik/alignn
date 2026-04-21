from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from alignn.data.dataset import JarvisGraphDataset, collate_graph_samples
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
