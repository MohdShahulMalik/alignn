"""Line graph builder for ALIGNN - computes bond angles from atomistic graph."""

from __future__ import annotations

import dgl
import torch
from typing import Optional


def compute_bond_cosines(edges: dgl.DGLEdgeData) -> dict[str, torch.Tensor]:
    """Compute bond angle cosines from bond displacement vectors.

    For line graph edge connecting bonds (a->b) and (b->c):
    - r1 = vector from a to b (src bond)
    - r2 = vector from b to c (dst bond)
    - cos(theta) = (r1 · r2) / (||r1|| ||r2||)

    Args:
        edges: DGL edge batch with source and destination node features

    Returns:
        Dictionary with "h" key containing bond angle cosine tensor
    """
    r1 = -edges.src["r"]
    r2 = edges.dst["r"]

    r1_norm = torch.norm(r1, dim=1)
    r2_norm = torch.norm(r2, dim=1)

    bond_cosine = torch.sum(r1 * r2, dim=1) / (r1_norm * r2_norm + 1e-8)
    bond_cosine = torch.clamp(bond_cosine, -1, 1)

    return {"h": bond_cosine}


def build_line_graph(
    g: dgl.DGLGraph,
    compute_angles: bool = True,
) -> dgl.DGLGraph:
    """Build line graph from atomistic graph.

    In the line graph:
    - Each node represents a bond (edge) from the original graph
    - Two nodes are connected if their corresponding bonds share an atom
    - This represents bond angle relationships (3-body interactions)

    Args:
        g: Atomistic DGL graph
        compute_angles: Whether to compute bond angle cosines

    Returns:
        Line graph with angle features on edges
    """
    lg = g.line_graph(shared=True)

    if compute_angles:
        lg.apply_edges(compute_bond_cosines)

    return lg


def build_graph_with_line_graph(
    atom_graph: dgl.DGLGraph,
    compute_angles: bool = True,
) -> tuple[dgl.DGLGraph, dgl.DGLGraph]:
    """Build both atom graph and line graph.

    Args:
        atom_graph: Atomistic graph from build_dgl_graph
        compute_angles: Whether to compute bond angle cosines

    Returns:
        Tuple of (atom_graph, line_graph)
    """
    lg = build_line_graph(atom_graph, compute_angles)
    return atom_graph, lg