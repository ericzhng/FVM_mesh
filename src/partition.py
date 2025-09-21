import os
from typing import Dict, List, Optional, Set

import numpy as np
import warnings

os.environ["METIS_DLL"] = os.path.join(os.getcwd(), "dll", "metis.dll")
try:
    import metis
except ImportError:
    metis = None

from src.mesh import Mesh


class PartitionResult:
    """Holds the result of a mesh partition, providing access to halo data."""

    def __init__(self, mesh: Mesh, parts: np.ndarray):
        if not mesh._is_analyzed:
            mesh.analyze_mesh()
        self.cell_neighbors = mesh.cell_neighbors

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
        """
        Builds communication maps for halo exchange between partitions.

        This method creates a set of mappings for each partition (rank) that
        enables efficient, index-based halo data exchange using MPI. It avoids
        costly searches by pre-calculating the local indices for sending owned
        cells and receiving halo cells.

        The output structure for each rank includes:
        - 'owned_cells': A list of global cell IDs owned by this rank.
        - 'halo_cells': A list of global cell IDs that make up the halo layer for this rank.
        - 'send': A dict mapping a neighbor rank to a list of local indices
                  (into 'owned_cells') of cells to be sent to that neighbor.
        - 'recv': A dict mapping a neighbor rank to a list of local indices
                  (into 'halo_cells') where incoming data from that neighbor
                  should be stored.

        The send and receive lists are ordered to ensure symmetric communication.
        """
        # 1. Globally determine which cells are sent between which partitions.
        #    `global_send_map[p][q]` will be a sorted list of global cell IDs
        #    that partition `p` must send to partition `q`.
        send_candidates: Dict[int, Dict[int, Set[int]]] = {
            r: {} for r in range(self.n_parts)
        }
        for g_idx, neighbors in enumerate(self.cell_neighbors):
            owner_part = self._parts[g_idx]
            for neighbor_g_idx in neighbors:
                if neighbor_g_idx != -1:
                    neighbor_part = self._parts[neighbor_g_idx]
                    if owner_part != neighbor_part:
                        # `g_idx` is on the boundary, so `owner_part` must send it
                        # to `neighbor_part`.
                        send_candidates[owner_part].setdefault(
                            int(neighbor_part), set()
                        ).add(g_idx)

        global_send_map: Dict[int, Dict[int, List[int]]] = {
            r: {} for r in range(self.n_parts)
        }
        for rank, send_dict in send_candidates.items():
            for neighbor_rank, g_indices_set in send_dict.items():
                # Sort for a canonical, ordered communication plan.
                global_send_map[rank][neighbor_rank] = sorted(list(g_indices_set))

        # 2. For each partition, build the final local index-based maps.
        out: Dict[int, Dict] = {}
        for rank in range(self.n_parts):
            # a. Define local-to-global mapping for owned cells.
            owned_cells_g = np.where(self._parts == rank)[0]
            g2l_owned_map = {g: l for l, g in enumerate(owned_cells_g)}

            # b. Build the halo layer and its local-to-global mapping.
            #    The halo is composed of cells this rank receives from its neighbors.
            #    The order is made deterministic by sorting neighbor ranks.
            halo_cells_g: List[int] = []
            halo_from_neighbors_g: Dict[int, List[int]] = {}

            all_neighbors = set(global_send_map[rank].keys())
            for r, send_map in global_send_map.items():
                if rank in send_map:
                    all_neighbors.add(r)

            for neighbor_rank in sorted(list(all_neighbors)):
                cells_to_recv = global_send_map.get(neighbor_rank, {}).get(rank)
                if cells_to_recv:
                    # `cells_to_recv` is already sorted from the global plan.
                    halo_from_neighbors_g[neighbor_rank] = cells_to_recv
                    halo_cells_g.extend(cells_to_recv)

            g2l_halo_map = {g: l for l, g in enumerate(halo_cells_g)}

            # c. Build the `send` map using local indices of owned cells.
            send_map_local: Dict[int, List[int]] = {}
            my_sends_g = global_send_map.get(rank, {})
            for neighbor_rank, g_indices_to_send in my_sends_g.items():
                send_map_local[neighbor_rank] = [
                    g2l_owned_map[g] for g in g_indices_to_send
                ]

            # d. Build the `recv` map using local indices of halo cells.
            recv_map_local: Dict[int, List[int]] = {}
            for neighbor_rank, g_indices_to_recv in halo_from_neighbors_g.items():
                # The order of received data matches the order of `g_indices_to_recv`.
                # We map each incoming item to its place in the local halo array.
                recv_map_local[neighbor_rank] = [
                    g2l_halo_map[g] for g in g_indices_to_recv
                ]

            out[rank] = {
                "owned_cells": owned_cells_g.tolist(),
                "halo_cells": halo_cells_g,
                "send": send_map_local,
                "recv": recv_map_local,
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

        if method == "metis":
            adjacency = _get_adjacency(mesh)
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
    """Partitions the mesh using a sequential coordinate bisection method."""
    # Check if n_parts is a power of two
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

    for i in range(1, n_parts):
        part_counts = np.bincount(parts)
        p_to_split = np.argmax(part_counts)

        idxs_to_split = np.where(parts == p_to_split)[0]

        if not idxs_to_split.any():
            continue

        pts = centroids[idxs_to_split]
        spans = pts.max(axis=0) - pts.min(axis=0)
        axis = int(np.argmax(spans))
        order = np.argsort(pts[:, axis])

        w = weights[idxs_to_split][order]
        cum = np.cumsum(w)
        total = cum[-1]

        if total == 0:
            split = len(order) // 2
        else:
            split = int(np.searchsorted(cum, total / 2.0))

        if split == 0 or split == len(order):
            split = len(order) // 2

        right_indices = idxs_to_split[order[split:]]
        parts[right_indices] = i

    return parts
