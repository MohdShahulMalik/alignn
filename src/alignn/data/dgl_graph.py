"""Convert graph dict to DGL graph for ALIGNN."""

from __future__ import annotations

import dgl
import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class CrystalGraph:
    """Container for crystal graph data."""
    g: dgl.DGLGraph
    lg: Optional[dgl.DGLGraph] = None
    lattice: Optional[torch.Tensor] = None


def build_dgl_graph(
    graph_dict: dict,
    compute_line_graph: bool = False,
) -> CrystalGraph:
    """Build a DGL graph from the graph dictionary.

    Args:
        graph_dict: Dictionary from graph_builder.py
        compute_line_graph: Whether to build line graph

    Returns:
        CrystalGraph with DGL graphs
    """
    num_nodes = graph_dict["num_nodes"]
    edge_src = graph_dict["edge_src"]
    edge_dst = graph_dict["edge_dst"]
    edge_r = graph_dict["edge_r"]
    edge_distance = graph_dict["edge_distance"]
    atomic_numbers = graph_dict["atomic_numbers"]
    positions = graph_dict["positions"]
    lattice_matrix = graph_dict["lattice_matrix"]

    g = dgl.graph((edge_src, edge_dst), num_nodes=num_nodes)

    g.ndata["atomic_number"] = torch.tensor(atomic_numbers, dtype=torch.long)
    g.ndata["pos"] = torch.tensor(positions, dtype=torch.float32)
    g.ndata["frac_coords"] = torch.tensor(
        _cart_to_frac(positions, lattice_matrix), dtype=torch.float32
    )

    g.edata["r"] = torch.tensor(edge_r, dtype=torch.float32)
    g.edata["d"] = torch.tensor(edge_distance, dtype=torch.float32)
    g.edata["images"] = torch.tensor(
        graph_dict["edge_image"], dtype=torch.float32
    )

    volume = abs(np.linalg.det(lattice_matrix))
    g.ndata["V"] = torch.tensor([volume] * num_nodes, dtype=torch.float32)

    lg = None
    if compute_line_graph:
        lg = g.line_graph(shared=True)
        lg.apply_edges(compute_bond_cosines)

    lattice = torch.tensor(lattice_matrix, dtype=torch.float32)

    return CrystalGraph(g=g, lg=lg, lattice=lattice)


def _cart_to_frac(cart_coords: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    """Convert Cartesian to fractional coordinates."""
    inv_lattice = np.linalg.inv(lattice)
    return cart_coords @ inv_lattice.T


def compute_bond_cosines(edges: dgl.DGLEdgeBatch) -> dict[str, torch.Tensor]:
    """Compute bond angle cosines for line graph edges."""
    r1 = -edges.src["r"]
    r2 = edges.dst["r"]

    r1_norm = torch.norm(r1, dim=1)
    r2_norm = torch.norm(r2, dim=1)

    bond_cosine = torch.sum(r1 * r2, dim=1) / (r1_norm * r2_norm + 1e-8)
    bond_cosine = torch.clamp(bond_cosine, -1, 1)

    return {"h": bond_cosine}


def collate_batch(
    samples: list[CrystalGraph],
) -> tuple[torch.Tensor, list[dgl.DGLGraph], list[torch.Tensor]]:
    """Collate function for DataLoader."""
    graphs = [s.g for s in samples]
    lattices = [s.lattice for s in samples]

    batched_g = dgl.batch(graphs)
    batched_lattices = torch.stack(lattices)

    return batched_g, batched_lattices


def collate_batch_with_line_graph(
    samples: list[CrystalGraph],
) -> tuple:
    """Collate function for ALIGNN with line graphs."""
    graphs = [s.g for s in samples]
    line_graphs = [s.lg for s in samples]
    lattices = [s.lattice for s in samples]

    batched_g = dgl.batch(graphs)
    batched_lg = dgl.batch(line_graphs)
    batched_lattices = torch.stack(lattices)

    return batched_g, batched_lg, batched_lattices
