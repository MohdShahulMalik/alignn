from __future__ import annotations

import dgl
import dgl.function as fn
import torch
import torch.nn as nn

from alignn.data.features import CrystalFeatureEncoder


class EdgeGatedGraphConv(nn.Module):
    """Minimal edge-gated message passing for the day 6 baseline."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.src_proj = nn.Linear(hidden_dim, hidden_dim)
        self.edge_proj = nn.Linear(hidden_dim, hidden_dim)
        self.edge_gate = nn.Linear(hidden_dim * 3, hidden_dim)
        self.node_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_update = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.edge_norm = nn.LayerNorm(hidden_dim)

    def _edge_messages(self, edges: dgl.udf.EdgeBatch) -> dict[str, torch.Tensor]:
        edge_input = torch.cat([edges.src["h"], edges.dst["h"], edges.data["e"]], dim=-1)
        gate = torch.sigmoid(self.edge_gate(edge_input))
        msg = gate * (self.src_proj(edges.src["h"]) + self.edge_proj(edges.data["e"]))
        edge_update = self.edge_update(edge_input)
        return {"m": msg, "e_update": edge_update}

    def forward(
        self,
        g: dgl.DGLGraph,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with g.local_scope():
            g.ndata["h"] = node_feats
            g.edata["e"] = edge_feats
            g.apply_edges(self._edge_messages)
            g.update_all(fn.copy_e("m", "m"), fn.sum("m", "agg"))

            node_input = torch.cat([node_feats, g.ndata["agg"]], dim=-1)
            node_out = self.node_norm(node_feats + self.node_update(node_input))
            edge_out = self.edge_norm(edge_feats + g.edata["e_update"])

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
