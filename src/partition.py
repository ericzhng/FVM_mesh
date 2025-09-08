import os
from typing import Dict, List, Optional, Set

import numpy as np

os.environ["METIS_DLL"] = os.path.join(os.getcwd(), "dll", "metis.dll")
try:
    import metis
except ImportError:
    metis = None

from src.mesh import Mesh


class PartitionResult:
    """Holds the result of a mesh partition, providing access to halo data."""

    def __init__(self, mesh: Mesh, parts: np.ndarray):
        self._mesh = mesh
        self._parts = parts
        self._halo_indices: Optional[Dict] = None

    @property
    def parts(self) -> np.ndarray:
        """The partition ID for each cell."""
        return self._parts

    @property
    def n_parts(self) -> int:
        """The number of partitions."""
        if self._parts.size == 0:
            return 0
        return int(np.max(self._parts) + 1)

    @property
    def halo_indices(self) -> Dict:
        """Lazily computes and returns halo indices for data exchange."""
        if self._halo_indices is None:
            self._halo_indices = self._build_halo_indices()
        return self._halo_indices

    def print_summary(self):
        """Prints a summary of the cell distribution across partitions."""
        if self.n_parts == 0:
            print("--- Partition summary ---")
            print("No partitions found.")
            return

        counts = np.bincount(self._parts, minlength=self.n_parts)
        print("--- Partition summary ---")
        print(f"Parts: {self.n_parts}")
        for p in range(self.n_parts):
            print(f" part {p}: cells = {counts[p]}")

    def _build_halo_indices(self) -> Dict:
        if not self._mesh._is_analyzed:
            self._mesh.analyze_mesh()

        out: Dict[int, Dict] = {}
        for rank in range(self.n_parts):
            owned_cell_indices = np.where(self._parts == rank)[0]
            owned_cells_global_to_local = {
                g_idx: l_idx for l_idx, g_idx in enumerate(owned_cell_indices)
            }

            halo_cells_by_neighbor_rank: Dict[int, Set[int]] = {}
            for cell_idx in owned_cell_indices:
                for neighbor_cell_idx in self._mesh.cell_neighbors[cell_idx]:
                    if neighbor_cell_idx != -1:
                        neighbor_part = int(self._parts[neighbor_cell_idx])
                        if neighbor_part != rank:
                            halo_cells_by_neighbor_rank.setdefault(
                                neighbor_part, set()
                            ).add(neighbor_cell_idx)

            send_map: Dict[int, List[int]] = {}
            recv_map: Dict[int, List[int]] = {}

            for neighbor_rank, halo_cells in halo_cells_by_neighbor_rank.items():
                recv_map[neighbor_rank] = sorted(list(halo_cells))

                cells_to_send: Set[int] = set()
                for halo_cell in halo_cells:
                    for neighbor_of_halo in self._mesh.cell_neighbors[halo_cell]:
                        if (
                            neighbor_of_halo != -1
                            and self._parts[neighbor_of_halo] == rank
                        ):
                            cells_to_send.add(neighbor_of_halo)

                send_map[neighbor_rank] = sorted(
                    [owned_cells_global_to_local[g_idx] for g_idx in cells_to_send]
                )

            out[rank] = {
                "owned_cells": owned_cell_indices.tolist(),
                "send": send_map,
                "recv": recv_map,
            }

        return out


def partition_mesh(
    mesh: Mesh,
    n_parts: int,
    method: str = "metis",
    cell_weights: Optional[np.ndarray] = None,
) -> PartitionResult:
    """Partition mesh elements into n_parts.

    Args:
        mesh: The mesh object to partition.
        n_parts: The number of partitions.
        method: The partitioning method ('metis', 'hierarchical').
        cell_weights: Optional weights for each cell.

    Returns:
        A PartitionResult object containing the partition data.
    """
    if n_parts <= 1:
        parts = np.zeros(mesh.num_cells, dtype=int)
    else:
        if not mesh._is_analyzed:
            mesh.analyze_mesh()

        adjacency = _get_adjacency(mesh)

        if method == "metis":
            parts = _partition_with_metis(adjacency, n_parts, cell_weights)
        elif method == "hierarchical":
            parts = _partition_with_hierarchical(mesh, n_parts, cell_weights)
        else:
            raise NotImplementedError(f"Partition method '{method}' not implemented")

    return PartitionResult(mesh, parts)


def _get_adjacency(mesh: Mesh) -> List[List[int]]:
    """Computes the adjacency list for the mesh cells."""
    adjacency: List[List[int]] = []
    for i in range(mesh.num_cells):
        neighs = {int(nb) for nb in mesh.cell_neighbors[i] if nb != -1}
        neighs.discard(i)
        adjacency.append(list(neighs))
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
