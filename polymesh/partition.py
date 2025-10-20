# -*- coding: utf-8 -*-
import os
from typing import List, Optional

import numpy as np
import warnings

from polymesh.core_mesh import CoreMesh

# Note: The METIS library is loaded dynamically. The path to the METIS DLL
# is set here. This might need to be adjusted depending on the user's
# environment.
os.environ["METIS_DLL"] = os.path.join(os.getcwd(), "dll", "metis.dll")
try:
    import metis
except ImportError:
    metis = None


def partition_mesh(
    mesh: CoreMesh,
    n_parts: int,
    method: str = "metis",
    cell_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Partition mesh elements into a specified number of parts.

    This function supports different partitioning methods, including METIS and
    a simple hierarchical coordinate bisection method.

    Args:
        mesh: The mesh object to partition.
        n_parts: The number of partitions.
        method: The partitioning method ('metis' or 'hierarchical').
        cell_weights: Optional weights for each cell.

    Returns:
        A numpy array of partition IDs for each cell.
    """
    if n_parts <= 1:
        return np.zeros(mesh.num_cells, dtype=int)

    if method == "metis":
        if not mesh.cell_neighbors.any():
            mesh._extract_cell_faces()
            mesh._extract_cell_neighbors()
        adjacency = _get_adjacency(mesh)
        parts = _partition_with_metis(adjacency, n_parts, cell_weights)
    elif method == "hierarchical":
        if not mesh.cell_centroids.any():
            mesh._compute_centroids()
        parts = _partition_with_hierarchical(mesh, n_parts, cell_weights)
    else:
        raise NotImplementedError(f"Partition method '{method}' not implemented")

    return parts


def _get_adjacency(mesh: CoreMesh) -> List[List[int]]:
    """
    Computes the adjacency list for the mesh cells.

    The adjacency list is a list of lists, where each inner list contains the
    indices of the neighbors of a cell.

    Args:
        mesh: The mesh object.

    Returns:
        The adjacency list.
    """
    adjacency: List[List[int]] = []
    for i in range(mesh.num_cells):
        neighs = {int(nb) for nb in mesh.cell_neighbors[i] if nb != -1}
        neighs.discard(i)
        adjacency.append(list(neighs))
    return adjacency


def _partition_with_metis(
    adjacency: List[List[int]], n_parts: int, cell_weights: Optional[np.ndarray]
) -> np.ndarray:
    """Partitions the mesh using the METIS library."""
    if metis is None:
        raise ImportError("METIS python binding not available")

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
    mesh: CoreMesh, n_parts: int, cell_weights: Optional[np.ndarray]
) -> np.ndarray:
    """Partitions the mesh using a sequential coordinate bisection method."""
    is_power_of_two = (n_parts > 0) and (n_parts & (n_parts - 1) == 0)
    if not is_power_of_two:
        warnings.warn(
            f"The 'hierarchical' partitioning method works best with a number of partitions "
            f"that is a power of two. You provided n_parts={n_parts}, which may result in "
            f"unevenly sized partitions."
        )

    centroids = mesh.cell_centroids
    weights = (
        cell_weights
        if cell_weights is not None
        else np.ones(mesh.num_cells, dtype=float)
    )
    parts = np.zeros(mesh.num_cells, dtype=int)

    # Iteratively bisect the largest partition until the desired number of
    # partitions is reached.
    for i in range(1, n_parts):
        # Find the partition with the most cells to split next.
        part_counts = np.bincount(parts)
        p_to_split = np.argmax(part_counts)
        idxs_to_split = np.where(parts == p_to_split)[0]

        if not idxs_to_split.any():
            continue

        # Determine the axis to split along by finding the longest dimension
        # of the bounding box of the partition's cell centroids.
        pts = centroids[idxs_to_split]
        spans = pts.max(axis=0) - pts.min(axis=0)
        axis = int(np.argmax(spans))
        order = np.argsort(pts[:, axis])

        # Find the median split point, taking cell weights into account.
        w = weights[idxs_to_split][order]
        cum = np.cumsum(w)
        total = cum[-1]

        if total == 0:
            split = len(order) // 2
        else:
            split = int(np.searchsorted(cum, total / 2.0))

        # Handle cases where the split is at the beginning or end.
        if split == 0 or split == len(order):
            split = len(order) // 2

        # Assign the new partition ID to the cells on one side of the split.
        right_indices = idxs_to_split[order[split:]]
        parts[right_indices] = i

    return parts


def print_partition_summary(parts: np.ndarray):
    """Prints a summary of the cell distribution across partitions."""
    if parts.size == 0:
        print("--- Partition summary ---")
        print("No partitions found.")
        return

    n_parts = int(np.max(parts) + 1)
    counts = np.bincount(parts, minlength=n_parts)

    print("--- Partition summary ---")
    print(f"Number of partitions: {n_parts}")
    for p in range(n_parts):
        print(f"  Partition {p}: {counts[p]} cells")
