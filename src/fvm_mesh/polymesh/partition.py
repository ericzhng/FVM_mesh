# -*- coding: utf-8 -*-
"""
Mesh partitioning tools.

This module provides functions for partitioning an unstructured mesh into multiple
subdomains. Partitioning is a crucial step for parallel processing of large
meshes, as it allows the computational load to be distributed across multiple
processors.

Key Features
------------
- Support for different partitioning methods, including METIS and a
  hierarchical coordinate bisection method.
- Ability to use cell weights to balance the partitioning.
- A simple interface for partitioning a `PolyMesh` object.

Functions
---------
:py:func:`partition_mesh`:
    Partitions a mesh into a specified number of parts.
:py:func:`print_partition_summary`:
    Prints a summary of the cell distribution across partitions.
"""

from __future__ import annotations

import os
import warnings
from typing import List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .poly_mesh import PolyMesh

# Note: The METIS library is loaded dynamically. The path to the METIS DLL
# is set here. This might need to be adjusted depending on the user's
# environment.
import os
import sys
from importlib import resources

# Note: The METIS library is loaded dynamically. The path to the METIS DLL
# is set here. This might need to be adjusted depending on the user's
# environment.
try:
    if sys.version_info < (3, 9):
        with resources.path("fvm_mesh", "dll") as dll_dir:
            os.environ["METIS_DLL"] = os.path.join(dll_dir, "metis.dll")
    else:
        dll_path = resources.files("fvm_mesh").joinpath("dll", "metis.dll")
        os.environ["METIS_DLL"] = str(dll_path)
except (ModuleNotFoundError, FileNotFoundError):
    # Fallback for when the package is not installed, e.g., when running tests directly
    os.environ["METIS_DLL"] = os.path.join(
        os.path.dirname(__file__), "..", "dll", "metis.dll"
    )


try:
    import metis
except ImportError:
    metis = None


def partition_mesh(
    mesh: PolyMesh,
    n_parts: int,
    method: str = "metis",
    cell_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Partitions mesh elements into a specified number of parts.

    This function supports different partitioning methods, including METIS and a
    simple hierarchical coordinate bisection method.

    Args:
        mesh: The mesh object to partition.
        n_parts: The number of partitions.
        method: The partitioning method ('metis' or 'hierarchical').
        cell_weights: Optional weights for each cell.

    Returns:
        A numpy array of partition IDs for each cell.
    """
    if n_parts <= 1:
        return np.zeros(mesh.n_cells, dtype=int)

    if method == "metis":
        return _partition_with_metis(mesh, n_parts, cell_weights)
    elif method == "hierarchical":
        return _partition_with_hierarchical(mesh, n_parts, cell_weights)
    else:
        raise NotImplementedError(f"Partition method '{method}' not implemented")


def _get_adjacency(mesh: PolyMesh) -> List[List[int]]:
    """
    Computes the adjacency list for the mesh cells.

    Args:
        mesh: The mesh object.

    Returns:
        The adjacency list.
    """
    if mesh.cell_neighbors.size == 0:
        mesh._extract_cell_faces()
        # compute topology (neighbors and tags)
        mesh._compute_face_topology()

    adjacency: List[List[int]] = []
    for i in range(mesh.n_cells):
        neighs = {int(nb) for nb in mesh.cell_neighbors[i] if nb != -1}
        neighs.discard(i)
        adjacency.append(list(neighs))
    return adjacency


def _partition_with_metis(
    mesh: PolyMesh, n_parts: int, cell_weights: Optional[np.ndarray]
) -> np.ndarray:
    """Partitions the mesh using the METIS library."""
    if metis is None:
        raise ImportError("METIS python binding not available")

    adjacency = _get_adjacency(mesh)
    try:
        vwgt = cell_weights.tolist() if cell_weights is not None else None
        _, parts = metis.part_graph(
            adjacency, nparts=n_parts, tpwgts=vwgt, recursive=True
        )
        return np.array(parts, dtype=int)
    except Exception as ex:
        raise RuntimeError(f"METIS partitioning failed: {ex}")


def _partition_with_hierarchical(
    mesh: PolyMesh, n_parts: int, cell_weights: Optional[np.ndarray]
) -> np.ndarray:
    """Partitions the mesh using a sequential coordinate bisection method."""
    is_power_of_two = (n_parts > 0) and (n_parts & (n_parts - 1) == 0)
    if not is_power_of_two:
        warnings.warn(
            f"The 'hierarchical' method works best with a power-of-two number of partitions. "
            f"Provided n_parts={n_parts} may result in uneven partitions."
        )

    if mesh.cell_centroids.size == 0:
        mesh._compute_cell_centroids()

    centroids = mesh.cell_centroids
    weights = cell_weights if cell_weights is not None else np.ones(mesh.n_cells)
    parts = np.zeros(mesh.n_cells, dtype=int)

    # Iteratively bisect the largest partition until the desired number of
    # partitions is reached.
    for i in range(1, n_parts):
        # Find the partition with the most cells to split next.
        part_counts = np.bincount(parts)
        p_to_split = np.argmax(part_counts)
        idxs_to_split = np.where(parts == p_to_split)[0]

        if idxs_to_split.size == 0:
            continue

        # Determine the axis to split along by finding the longest dimension
        # of the bounding box of the partition's cell centroids.
        pts = centroids[idxs_to_split]
        axis = int(np.argmax(pts.max(axis=0) - pts.min(axis=0)))
        order = np.argsort(pts[:, axis])

        # Find the median split point, taking cell weights into account.
        w = weights[idxs_to_split][order]
        cum_w = np.cumsum(w)
        total_w = cum_w[-1]

        split_idx = len(order) // 2
        if total_w > 0:
            split_idx = int(np.searchsorted(cum_w, total_w / 2.0))

        # Handle cases where the split is at the beginning or end.
        if split_idx == 0 or split_idx == len(order):
            split_idx = len(order) // 2

        # Assign the new partition ID to the cells on one side of the split.
        right_indices = idxs_to_split[order[split_idx:]]
        parts[right_indices] = i

    return parts


def print_partition_summary(parts: np.ndarray) -> None:
    """Prints a summary of the cell distribution across partitions."""
    if parts.size == 0:
        print("--- Partition Summary ---")
        print("No partitions found.")
        return

    n_parts = int(np.max(parts) + 1)
    counts = np.bincount(parts, minlength=n_parts)

    print("--- Partition Summary ---")
    print(f"Number of partitions: {n_parts}")
    for p, count in enumerate(counts):
        print(f"  Partition {p}: {count} cells")
