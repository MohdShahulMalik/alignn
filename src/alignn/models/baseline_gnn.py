from __future__ import annotations

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F

from alignn.data.features import CrystalFeatureEncoder


class EdgeGatedGraphConv(nn.Module):
    """Original ALIGNN-style normalized edge-gated graph convolution."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.src_gate = nn.Linear(hidden_dim, hidden_dim)
        self.dst_gate = nn.Linear(hidden_dim, hidden_dim)
        self.edge_gate = nn.Linear(hidden_dim, hidden_dim)
        self.src_update = nn.Linear(hidden_dim, hidden_dim)
        self.dst_update = nn.Linear(hidden_dim, hidden_dim)
        self.node_norm = nn.BatchNorm1d(hidden_dim)
        self.edge_norm = nn.BatchNorm1d(hidden_dim)

    def forward(
        self,
        g: dgl.DGLGraph,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with g.local_scope():
            g.ndata["e_src"] = self.src_gate(node_feats)
            g.ndata["e_dst"] = self.dst_gate(node_feats)
            g.apply_edges(fn.u_add_v("e_src", "e_dst", "e_nodes"))
            edge_update = g.edata.pop("e_nodes") + self.edge_gate(edge_feats)

            g.edata["sigma"] = torch.sigmoid(edge_update)
            g.ndata["dst_update"] = self.dst_update(node_feats)
            g.update_all(
                fn.u_mul_e("dst_update", "sigma", "m"),
                fn.sum("m", "sum_sigma_h"),
            )
            g.update_all(fn.copy_e("sigma", "m"), fn.sum("m", "sum_sigma"))
            aggregate = g.ndata["sum_sigma_h"] / (g.ndata["sum_sigma"] + 1e-6)

            node_delta = self.src_update(node_feats) + aggregate
            node_out = node_feats + F.silu(self.node_norm(node_delta))
            edge_out = edge_feats + F.silu(self.edge_norm(edge_update))

        return node_out, edge_out


class BaselineGNN(nn.Module):
    """Distance-aware atom graph baseline without line-graph updates."""

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 4,
        num_elements: int = 100,
        bond_rbf_bins: int = 80,
    ) -> None:
        super().__init__()
        self.encoder = CrystalFeatureEncoder(
            num_elements=num_elements,
            rbf_bins=bond_rbf_bins,
            angle_bins=40,
            embedding_dim=hidden_dim,
        )
        self.layers = nn.ModuleList(
            [EdgeGatedGraphConv(hidden_dim=hidden_dim) for _ in range(num_layers)]
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, g: dgl.DGLGraph) -> torch.Tensor:
        node_feats, edge_feats, _ = self.encoder(g)

        for layer in self.layers:
            node_feats, edge_feats = layer(g, node_feats, edge_feats)

        with g.local_scope():
            g.ndata["h"] = node_feats
            graph_feats = dgl.mean_nodes(g, "h")

        return self.readout(graph_feats).squeeze(-1)
