from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from pymatgen.core import Structure


@dataclass
class NeighborEdge:
    """
    A dataclass to represent an edge between two neighboring atoms in a crystal structure.
    """

    src: int
    dst: int
    distance: float
    r: np.ndarray
    image: tuple[int, int, int]


def _neighbor_image(neighbor) -> tuple[int, int, int]:
    """
    Extract the image information from a neighbor dictionary.
    """
    image = getattr(neighbor, "image", None)
    if image is not None and len(image) == 3:
        a, b, c = image
        return int(a), int(b), int(c)
    index = getattr(neighbor, "index", None)
    if isinstance(index, (tuple, list)) and len(index) == 3:
        a, b, c = index
        return int(a), int(b), int(c)

    return (0, 0, 0)


def _collect_atom_neighbors(
    structure: Structure,
    cutoff: float,
    atom_index: int,
    max_neighbors: int,
    tolerance: float = 1e-5,
) -> list[NeighborEdge]:
    """
    Collect the neighboring atoms for each atom in the structure within a specified cutoff distance.
    """

    center = np.array(structure[atom_index].coords, dtype=float)
    raw_neighbors = structure.get_neighbors(structure[atom_index], cutoff)

    neighbors = []
    for neighbor in raw_neighbors:
        neighbor_coords = np.array(neighbor.coords, dtype=float)
        distance = float(np.linalg.norm(neighbor_coords - center))

        if distance <= cutoff + tolerance:
            r = neighbor_coords - center
            image = _neighbor_image(neighbor)
            neighbors.append(
                NeighborEdge(
                    src=atom_index,
                    dst=neighbor.index,
                    distance=distance,
                    r=r,
                    image=image,
                )
            )

    neighbors.sort(key=lambda x: x.distance)

    if len(neighbors) <= max_neighbors:
        return neighbors

    kth_distance = neighbors[max_neighbors - 1].distance
    kept_neighbors = [
        edge for edge in neighbors if edge.distance <= kth_distance + tolerance
    ]

    return kept_neighbors


def get_k_nearest_edges(
    structure: Structure,
    cutoff: float = 8.0,
    max_neighbors: int = 12,
    tolerance: float = 1e-5,
) -> list[NeighborEdge]:
    """
    Get the k-nearest neighbors for each atom in the structure.
    """
    all_neighbors = []
    for atom_index in range(len(structure)):
        neighbors = _collect_atom_neighbors(
            structure, cutoff, atom_index, max_neighbors, tolerance
        )
        all_neighbors.extend(neighbors)
    return all_neighbors


def build_atom_graph(
    structure: Structure,
    cutoff: float = 8.0,
    max_neighbors: int = 12,
    tolerance: float = 1e-5,
) -> dict:
    """
    Build the graph edges for the given structure based on the k-nearest neighbors.
    """
    edges = get_k_nearest_edges(structure, cutoff, max_neighbors, tolerance)

    atomic_numbers = np.array([site.specie.Z for site in structure], dtype=np.int64)

    positions = np.array([site.coords for site in structure], dtype=np.float32)

    edge_src = np.array([edge.src for edge in edges], dtype=np.int64)
    edge_dst = np.array([edge.dst for edge in edges], dtype=np.int64)
    edge_r = np.array([edge.r for edge in edges], dtype=np.float32)
    edge_distance = np.array([edge.distance for edge in edges], dtype=np.float32)
    edge_image = np.array([edge.image for edge in edges], dtype=np.int64)

    graph = {
        "num_nodes": int(len(structure)),
        "atomic_numbers": atomic_numbers,
        "positions": positions,
        "lattice_matrix": np.array(structure.lattice.matrix, dtype=np.float32),
        "edge_src": edge_src,
        "edge_dst": edge_dst,
        "edge_r": edge_r,
        "edge_distance": edge_distance,
        "edge_image": edge_image,
    }

    return graph


def validate_graph(graph: dict, tol: float = 1e-8) -> None:
    """Raise if graph contents are inconsistent."""
    num_nodes = int(graph["num_nodes"])
    edge_src = np.asarray(graph["edge_src"])
    edge_dst = np.asarray(graph["edge_dst"])
    edge_r = np.asarray(graph["edge_r"])
    edge_distance = np.asarray(graph["edge_distance"])

    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive.")

    if len(edge_src) != len(edge_dst):
        raise ValueError("edge_src and edge_dst length mismatch.")

    if len(edge_src) != len(edge_r):
        raise ValueError("edge arrays length mismatch.")

    if len(edge_src) != len(edge_distance):
        raise ValueError("edge distance length mismatch.")

    if len(edge_src) == 0:
        raise ValueError("Graph has no edges.")

    if edge_src.min() < 0 or edge_dst.min() < 0:
        raise ValueError("Negative node index in edges.")

    if edge_src.max() >= num_nodes or edge_dst.max() >= num_nodes:
        raise ValueError("Edge index exceeds node count.")

    if np.isnan(edge_r).any():
        raise ValueError("NaN found in edge displacement vectors.")

    if np.isnan(edge_distance).any():
        raise ValueError("NaN found in edge distances.")

    if (edge_distance <= tol).any():
        raise ValueError("Non-positive edge distance found.")


def graph_stats(graph: dict) -> dict:
    """
    Compute statistics about the graph, such as the number of nodes and edges.
    """
    num_nodes = graph["num_nodes"]
    num_edges = len(graph["edge_src"])

    if num_nodes == 0:
        raise ValueError("Graph must have at least one node to compute statistics.")

    if num_edges == 0:
        return {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "average_degree": 0.0,
            "min_distance": None,
            "max_distance": None,
        }

    edge_distance = np.asarray(graph["edge_distance"], dtype=float)
    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "average_degree": float(num_edges / num_nodes),
        "min_distance": float(edge_distance.min()),
        "max_distance": float(edge_distance.max()),
    }
