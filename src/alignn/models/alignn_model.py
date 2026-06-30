from __future__ import annotations

import dgl
import torch
import torch.nn as nn

from alignn.data.features import CrystalFeatureEncoder
from alignn.models.baseline_gnn import EdgeGatedGraphConv


class ALIGNNLayer(nn.Module):
    """Alternating atom-graph and line-graph update block.

    Atom graph edges are bonds. Line graph nodes are those same bonds, and line
    graph edges carry angle features between adjacent bonds.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.atom_graph_update = EdgeGatedGraphConv(hidden_dim=hidden_dim, dropout=dropout)
        self.line_graph_update = EdgeGatedGraphConv(hidden_dim=hidden_dim, dropout=dropout)

    def forward(
        self,
        g: dgl.DGLGraph,
        lg: dgl.DGLGraph,
        atom_feats: torch.Tensor,
        bond_feats: torch.Tensor,
        angle_feats: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        atom_feats, bond_feats = self.atom_graph_update(g, atom_feats, bond_feats)
        bond_feats, angle_feats = self.line_graph_update(lg, bond_feats, angle_feats)
        return atom_feats, bond_feats, angle_feats


class ALIGNNGraphEncoder(nn.Module):
    """Shared ALIGNN graph encoder that returns pooled crystal features."""

    def __init__(
        self,
        hidden_dim: int = 64,
        alignn_layers: int = 4,
        gcn_layers: int = 4,
        num_elements: int = 100,
        bond_rbf_bins: int = 80,
        angle_rbf_bins: int = 40,
        readout: str = "mean",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if readout not in {"mean", "meanmax"}:
            raise ValueError(f"Unsupported ALIGNN readout: {readout}")
        self.readout_name = readout
        self.encoder = CrystalFeatureEncoder(
            num_elements=num_elements,
            rbf_bins=bond_rbf_bins,
            angle_bins=angle_rbf_bins,
            embedding_dim=hidden_dim,
        )
        self.alignn_layers = nn.ModuleList(
            [ALIGNNLayer(hidden_dim=hidden_dim, dropout=dropout) for _ in range(alignn_layers)]
        )
        self.gcn_layers = nn.ModuleList(
            [EdgeGatedGraphConv(hidden_dim=hidden_dim, dropout=dropout) for _ in range(gcn_layers)]
        )
        readout_dim = hidden_dim * 2 if readout == "meanmax" else hidden_dim
        self.output_dim = readout_dim

    def forward(self, g: dgl.DGLGraph, lg: dgl.DGLGraph) -> torch.Tensor:
        atom_feats, bond_feats, angle_feats = self.encoder(g, lg)
        if angle_feats is None:
            raise ValueError("ALIGNN requires a line graph with angle features.")

        for layer in self.alignn_layers:
            atom_feats, bond_feats, angle_feats = layer(
                g,
                lg,
                atom_feats,
                bond_feats,
                angle_feats,
            )

        for layer in self.gcn_layers:
            atom_feats, bond_feats = layer(g, atom_feats, bond_feats)

        with g.local_scope():
            g.ndata["h"] = atom_feats
            graph_feats = dgl.mean_nodes(g, "h")
            if self.readout_name == "meanmax":
                graph_feats = torch.cat([graph_feats, dgl.max_nodes(g, "h")], dim=-1)

        return graph_feats


class ALIGNNModel(nn.Module):
    """ALIGNN-style graph regressor with bond-angle message passing."""

    def __init__(
        self,
        hidden_dim: int = 64,
        alignn_layers: int = 4,
        gcn_layers: int = 4,
        num_elements: int = 100,
        bond_rbf_bins: int = 80,
        angle_rbf_bins: int = 40,
        readout: str = "mean",
        energy_mult_natoms: bool = False,
        penalty_factor: float = 0.0,
        penalty_threshold: float = 1.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = ALIGNNGraphEncoder(
            hidden_dim=hidden_dim,
            alignn_layers=alignn_layers,
            gcn_layers=gcn_layers,
            num_elements=num_elements,
            bond_rbf_bins=bond_rbf_bins,
            angle_rbf_bins=angle_rbf_bins,
            readout=readout,
            dropout=dropout,
        )
        self.readout = nn.Linear(self.encoder.output_dim, 1)
        self.energy_mult_natoms = energy_mult_natoms
        self.penalty_factor = penalty_factor
        self.penalty_threshold = penalty_threshold

    def encode_graph(self, g: dgl.DGLGraph, lg: dgl.DGLGraph) -> torch.Tensor:
        return self.encoder(g, lg)

    def forward(self, g: dgl.DGLGraph, lg: dgl.DGLGraph) -> torch.Tensor:
        graph_feats = self.encode_graph(g, lg)
        out = self.readout(graph_feats).squeeze(-1)
        if self.energy_mult_natoms:
            num_atoms = g.batch_num_nodes().to(out.device).float()
            out = out * num_atoms
        if self.penalty_factor > 0:
            distances = g.edata["d"].reshape(-1).to(out.device)
            edge_penalties = self.penalty_factor * (
                self.penalty_threshold - distances
            ).clamp_min(0)
            batch_sizes = g.batch_num_edges().tolist()
            penalties = [
                penalty.sum()
                for penalty in torch.split(edge_penalties, batch_sizes)
            ]
            out = out + torch.stack(penalties).to(out.device)
        return out


class ZeroInflatedALIGNNModel(nn.Module):
    """ALIGNN encoder with a binary classifier head for zero-inflated regression.

    For targets like mbj_bandgap where most values are exactly zero, the classifier
    learns P(target > 0) and the regression head predicts the nonzero value.
    Final prediction: regression_pred * sigmoid(classifier_logit) — soft gating
    so even misclassified zeros are suppressed by low probability.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        alignn_layers: int = 4,
        gcn_layers: int = 4,
        num_elements: int = 100,
        bond_rbf_bins: int = 80,
        angle_rbf_bins: int = 40,
        readout: str = "mean",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = ALIGNNGraphEncoder(
            hidden_dim=hidden_dim,
            alignn_layers=alignn_layers,
            gcn_layers=gcn_layers,
            num_elements=num_elements,
            bond_rbf_bins=bond_rbf_bins,
            angle_rbf_bins=angle_rbf_bins,
            readout=readout,
            dropout=dropout,
        )
        self.regression_head = nn.Linear(self.encoder.output_dim, 1)
        self.classifier_head = nn.Linear(self.encoder.output_dim, 1)

    def forward(self, g: dgl.DGLGraph, lg: dgl.DGLGraph) -> tuple[torch.Tensor, torch.Tensor]:
        graph_feats = self.encoder(g, lg)
        regression_pred = self.regression_head(graph_feats).squeeze(-1)
        classifier_logit = self.classifier_head(graph_feats).squeeze(-1)
        return regression_pred, classifier_logit

    @torch.no_grad()
    def predict(self, g: dgl.DGLGraph, lg: dgl.DGLGraph) -> torch.Tensor:
        """Inference: soft-gate regression output by classifier probability."""
        regression_pred, classifier_logit = self.forward(g, lg)
        prob = torch.sigmoid(classifier_logit)
        return regression_pred.clamp_min(0) * prob


class MultiTaskALIGNNModel(nn.Module):
    """Shared ALIGNN encoder with one small regression head per target."""

    def __init__(
        self,
        target_names: list[str],
        hidden_dim: int = 64,
        alignn_layers: int = 4,
        gcn_layers: int = 4,
        num_elements: int = 100,
        bond_rbf_bins: int = 80,
        angle_rbf_bins: int = 40,
        readout: str = "mean",
        head_hidden_dim: int | None = None,
        head_configs: dict[str, dict] | None = None,
        head_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not target_names:
            raise ValueError("MultiTaskALIGNNModel requires at least one target.")
        self.target_names = list(target_names)
        self.target_to_id = {target: index for index, target in enumerate(self.target_names)}
        self.encoder = ALIGNNGraphEncoder(
            hidden_dim=hidden_dim,
            alignn_layers=alignn_layers,
            gcn_layers=gcn_layers,
            num_elements=num_elements,
            bond_rbf_bins=bond_rbf_bins,
            angle_rbf_bins=angle_rbf_bins,
            readout=readout,
        )
        default_head_dim = head_hidden_dim or hidden_dim
        self.head_norms = nn.ModuleDict(
            {target: nn.LayerNorm(self.encoder.output_dim) for target in self.target_names}
        )
        self.heads = nn.ModuleDict()
        for target in self.target_names:
            cfg = (head_configs or {}).get(target, {})
            h_dim = cfg.get("hidden_dim", default_head_dim)
            n_layers = cfg.get("n_layers", 2)
            dropout = cfg.get("dropout", head_dropout)
            layers: list[nn.Module] = []
            in_dim = self.encoder.output_dim
            for _ in range(n_layers - 1):
                layers.extend([
                    nn.Linear(in_dim, h_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                ])
                in_dim = h_dim
            layers.append(nn.Linear(in_dim, 1))
            self.heads[target] = nn.Sequential(*layers)

    def forward(
        self,
        g: dgl.DGLGraph,
        lg: dgl.DGLGraph,
        target_name: str,
    ) -> torch.Tensor:
        if target_name not in self.heads:
            raise KeyError(f"Unknown multi-task target: {target_name}")
        graph_feats = self.encoder(g, lg)
        graph_feats = self.head_norms[target_name](graph_feats)
        return self.heads[target_name](graph_feats).squeeze(-1)
