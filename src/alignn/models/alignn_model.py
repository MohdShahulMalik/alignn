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

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.atom_graph_update = EdgeGatedGraphConv(hidden_dim=hidden_dim)
        self.line_graph_update = EdgeGatedGraphConv(hidden_dim=hidden_dim)

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
            [ALIGNNLayer(hidden_dim=hidden_dim) for _ in range(alignn_layers)]
        )
        self.gcn_layers = nn.ModuleList(
            [EdgeGatedGraphConv(hidden_dim=hidden_dim) for _ in range(gcn_layers)]
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
        )
        self.readout = nn.Linear(self.encoder.output_dim, 1)

    def encode_graph(self, g: dgl.DGLGraph, lg: dgl.DGLGraph) -> torch.Tensor:
        return self.encoder(g, lg)

    def forward(self, g: dgl.DGLGraph, lg: dgl.DGLGraph) -> torch.Tensor:
        graph_feats = self.encode_graph(g, lg)
        return self.readout(graph_feats).squeeze(-1)


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
        head_dim = head_hidden_dim or hidden_dim
        self.heads = nn.ModuleDict(
            {
                target: nn.Sequential(
                    nn.Linear(self.encoder.output_dim, head_dim),
                    nn.SiLU(),
                    nn.Linear(head_dim, 1),
                )
                for target in self.target_names
            }
        )

    def forward(
        self,
        g: dgl.DGLGraph,
        lg: dgl.DGLGraph,
        target_name: str,
    ) -> torch.Tensor:
        if target_name not in self.heads:
            raise KeyError(f"Unknown multi-task target: {target_name}")
        graph_feats = self.encoder(g, lg)
        return self.heads[target_name](graph_feats).squeeze(-1)
