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
        self.readout = nn.Linear(readout_dim, 1)

    def forward(self, g: dgl.DGLGraph, lg: dgl.DGLGraph) -> torch.Tensor:
        atom_feats, bond_feats, angle_feats = self.encoder(g, lg)
        if angle_feats is None:
            raise ValueError("ALIGNNModel requires a line graph with angle features.")

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

        return self.readout(graph_feats).squeeze(-1)
