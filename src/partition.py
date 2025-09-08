import os
from typing import List, Optional

import numpy as np

os.environ["METIS_DLL"] = os.path.join(os.getcwd(), "dll", "metis.dll")
try:
    import metis  # type: ignore
except ImportError:
    metis = None

from src.mesh import Mesh


def partition_mesh(
    mesh: Mesh,
    n_parts: int,
    method: str = "metis",
    cell_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Partition mesh elements into n_parts.

    Args:
        mesh: The mesh object to partition.
        n_parts: The number of partitions.
        method: The partitioning method ('metis', 'hierarchical').
        cell_weights: Optional weights for each cell.

    Returns:
        An array of partition indices for each cell.
    """
    if n_parts <= 1:
        return np.zeros(mesh.num_cells, dtype=int)

    if not mesh._is_analyzed:
        mesh.analyze_mesh()

    adjacency = _get_adjacency(mesh)

    if method == "metis":
        return _partition_with_metis(adjacency, n_parts, cell_weights)
    elif method == "hierarchical":
        return _partition_with_hierarchical(mesh, n_parts, cell_weights)
    else:
        raise NotImplementedError(f"Partition method '{method}' not implemented")


def _get_adjacency(mesh: Mesh) -> List[List[int]]:
    """Computes the adjacency list for the mesh cells."""
    adjacency: List[List[int]] = []
    for i in range(mesh.num_cells):
        neighs = {int(nb) for nb in mesh.cell_neighbors[i] if nb != -1}
        neighs.discard(i)
        adjacency.append(sorted(neighs))
    return adjacency


def _partition_with_metis(
    adjacency: List[List[int]], n_parts: int, cell_weights: Optional[np.ndarray]
) -> np.ndarray:
    """Partitions the mesh using METIS."""
    if metis is None:
        raise ImportError("metis python binding not available")
    try:
        if cell_weights is not None:
            try:
                _, parts = metis.part_graph(
                    adjacency, nparts=n_parts, vwgt=cell_weights.tolist()
                )
            except TypeError:
                _, parts = metis.part_graph(
                    adjacency, nparts=n_parts, vweights=cell_weights.tolist()
                )
        else:
            _, parts = metis.part_graph(adjacency, nparts=n_parts)
        return np.array(parts, dtype=int)
    except Exception as ex:
        raise RuntimeError(f"METIS partitioning failed: {ex}")


def _partition_with_hierarchical(
    mesh: Mesh, n_parts: int, cell_weights: Optional[np.ndarray]
) -> np.ndarray:
    """Partitions the mesh using a hierarchical coordinate bisection method."""
    centroids = mesh.cell_centroids
    weights = (
        cell_weights
        if cell_weights is not None
        else np.ones(mesh.num_cells, dtype=float)
    )
    parts = -np.ones(mesh.num_cells, dtype=int)

    def recurse(idxs: np.ndarray, part_ids: List[int]):
        if len(part_ids) == 1:
            parts[idxs] = part_ids[0]
            return
        pts = centroids[idxs]
        spans = pts.max(axis=0) - pts.min(axis=0)
        axis = int(np.argmax(spans))
        order = np.argsort(pts[:, axis])
        w = weights[idxs][order]
        cum = np.cumsum(w)
        total = cum[-1]
        split = int(np.searchsorted(cum, total / 2.0))
        if split == 0 or split == len(order):
            split = len(order) // 2
        left = idxs[order[:split]]
        right = idxs[order[split:]]
        mid = len(part_ids) // 2
        recurse(left, part_ids[:mid])
        recurse(right, part_ids[mid:])

    all_idx = np.arange(mesh.num_cells, dtype=int)
    recurse(all_idx, list(range(n_parts)))
    return parts
