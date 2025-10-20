# -*- coding: utf-8 -*-
import copy
from typing import Dict, List, Set, Sequence, Optional

import numpy as np

from .reorder import renumber_cells, renumber_nodes
from .poly_mesh import PolyMesh
from .core_mesh import CoreMesh
from .partition import partition_mesh, print_partition_summary


def _find_send_candidates(
    mesh: CoreMesh, parts: np.ndarray
) -> Dict[int, Dict[int, List[int]]]:
    """
    Determines which cells each partition needs to send to its neighbors.
    """
    n_parts = int(np.max(parts) + 1) if parts.size > 0 else 0
    send_candidates: Dict[int, Dict[int, Set[int]]] = {r: {} for r in range(n_parts)}

    for g_idx, neighbors in enumerate(mesh.cell_neighbors):
        owner_part = parts[g_idx]
        for neighbor_g_idx in neighbors:
            if neighbor_g_idx != -1:
                neighbor_part = parts[neighbor_g_idx]
                if owner_part != neighbor_part:
                    send_candidates[owner_part].setdefault(
                        int(neighbor_part), set()
                    ).add(g_idx)

    # Convert sets to sorted lists for consistency
    global_send_map: Dict[int, Dict[int, List[int]]] = {r: {} for r in range(n_parts)}
    for rank, send_dict in send_candidates.items():
        for neighbor_rank, g_indices_set in send_dict.items():
            global_send_map[rank][neighbor_rank] = sorted(list(g_indices_set))

    return global_send_map


def _compute_halo_indices(mesh: CoreMesh, parts: np.ndarray) -> Dict:
    """
    Builds communication maps for halo exchange between partitions.
    """
    n_parts = int(np.max(parts) + 1) if parts.size > 0 else 0
    if n_parts <= 1:
        if mesh.num_cells > 0:
            return {
                0: {
                    "owned_cells": list(range(mesh.num_cells)),
                    "halo_cells": [],
                    "send": {},
                    "recv": {},
                }
            }
        return {}

    global_send_map = _find_send_candidates(mesh, parts)

    out: Dict[int, Dict] = {}
    for rank in range(n_parts):
        owned_cells_g = np.where(parts == rank)[0]
        g2l_owned_map = {g: l for l, g in enumerate(owned_cells_g)}

        halo_cells_g: List[int] = []
        halo_from_neighbors_g: Dict[int, List[int]] = {}

        all_neighbors = set(global_send_map.get(rank, {}).keys())
        for r, send_map in global_send_map.items():
            if rank in send_map:
                all_neighbors.add(r)

        for neighbor_rank in sorted(list(all_neighbors)):
            cells_to_recv = global_send_map.get(neighbor_rank, {}).get(rank)
            if cells_to_recv:
                halo_from_neighbors_g[neighbor_rank] = cells_to_recv
                halo_cells_g.extend(cells_to_recv)

        g2l_halo_map = {g: l for l, g in enumerate(halo_cells_g)}

        send_map_local: Dict[int, List[int]] = {}
        my_sends_g = global_send_map.get(rank, {})
        for neighbor_rank, g_indices_to_send in my_sends_g.items():
            send_map_local[neighbor_rank] = [
                g2l_owned_map[g] for g in g_indices_to_send
            ]

        recv_map_local: Dict[int, List[int]] = {}
        for neighbor_rank, g_indices_to_recv in halo_from_neighbors_g.items():
            recv_map_local[neighbor_rank] = [g2l_halo_map[g] for g in g_indices_to_recv]

        out[rank] = {
            "owned_cells": owned_cells_g.tolist(),
            "halo_cells": halo_cells_g,
            "send": send_map_local,
            "recv": recv_map_local,
        }

    return out


def _remap_connectivity(
    global_conn: Sequence[np.ndarray], g2l_map: Dict[int, int]
) -> List[List[int]]:
    """Remaps a global connectivity sequence to a local one."""
    local_conn = []
    for item_conn_g in global_conn:
        local_conn.append([g2l_map[g] for g in item_conn_g])
    return local_conn


class LocalMesh(PolyMesh):
    """
    Represents the mesh for a single partition (process) in a distributed setup.

    This class is central to parallel FVM computations. It holds the geometric
    and topological data for a subset of the global mesh, including cells owned
    by the current process and "halo" cells owned by neighboring processes.
    Halo cells are required for computing gradients and fluxes at partition
    boundaries.

    All entities (nodes, cells) are renumbered into a local, contiguous,
    0-based index space for computational efficiency. Crucially, this class
    maintains mappings (`l2g_cells`, `l2g_nodes`) to recover the original
    global indices, which is essential for assembling results from all
    partitions back into a global solution.

    The class provides methods to reorder cells and nodes, which can
    significantly improve cache efficiency and solver performance.

    Attributes:
        rank (int): The rank of the partition this mesh belongs to.
        num_owned_cells (int): The number of cells owned by this partition.
        num_halo_cells (int): The number of halo cells.
        l2g_cells (np.ndarray): Map from local cell ID to global cell ID.
        g2l_cells (Dict[int, int]): Map from global cell ID to local cell ID.
        l2g_nodes (np.ndarray): Map from local node ID to global node ID.
        g2l_nodes (Dict[int, int]): Map from global node ID to local node ID.
        send_map (Dict[int, List[int]]): Halo communication map for sending data.
        recv_map (Dict[int, List[int]]): Halo communication map for receiving data.
        use_reordered_cells (bool): Flag indicating if cells are reordered.
        use_reordered_nodes (bool): Flag indicating if nodes are reordered.
    """

    def __init__(
        self,
        rank: int,
        num_owned_cells: int,
        num_halo_cells: int,
        l2g_cells: np.ndarray,
        g2l_cells: Dict[int, int],
        l2g_nodes: np.ndarray,
        g2l_nodes: Dict[int, int],
        send_map: Dict[int, List[int]],
        recv_map: Dict[int, List[int]],
    ):
        super().__init__()
        self.rank = rank
        self.num_owned_cells = num_owned_cells
        self.num_halo_cells = num_halo_cells
        self.l2g_cells = l2g_cells
        self.g2l_cells = g2l_cells
        self.l2g_nodes = l2g_nodes
        self.g2l_nodes = g2l_nodes
        self.send_map = send_map
        self.recv_map = recv_map

        self.use_reordered_cells = False
        self.use_reordered_nodes = False

    @classmethod
    def from_global_mesh(cls, global_mesh: CoreMesh, parts: np.ndarray, rank: int, halo_info: Dict):
        """
        Factory method to construct a LocalMesh for a specific partition.

        This method encapsulates the complex process of extracting a partition's
        data from a global mesh, including owned cells, halo cells, and all
        necessary mappings and connectivity.

        Args:
            global_mesh: The complete, unpartitioned mesh.
            parts: The result from the partitioning process.
            rank: The ID of the partition for which to create the local mesh.
            halo_info: A dictionary containing the halo information for this rank.
        """
        if not global_mesh.cell_neighbors.size > 0:
            global_mesh.analyze_mesh()

        # 1. Initialize cell maps
        owned_cells_g = halo_info["owned_cells"]
        halo_cells_g = halo_info["halo_cells"]
        num_owned_cells = len(owned_cells_g)
        num_halo_cells = len(halo_cells_g)
        l2g_cells = np.array(owned_cells_g + halo_cells_g, dtype=int)
        g2l_cells = {g: l for l, g in enumerate(l2g_cells)}

        # 2. Initialize node maps
        local_cell_nodes_g = [global_mesh.cell_connectivity[g] for g in l2g_cells]
        unique_node_g_indices = np.unique(np.concatenate(local_cell_nodes_g))
        l2g_nodes = unique_node_g_indices
        g2l_nodes = {g: l for l, g in enumerate(l2g_nodes)}

        # 3. Build local mesh data
        node_coords = global_mesh.node_coords[l2g_nodes]
        cell_connectivity = _remap_connectivity(
            [global_mesh.cell_connectivity[g] for g in l2g_cells], g2l_nodes
        )
        if global_mesh.cell_type_ids.size > 0:
            cell_type_ids = global_mesh.cell_type_ids[l2g_cells]
        else:
            cell_type_ids = np.array([])

        # 4. Build local neighbors
        num_local_cells = len(cell_connectivity)
        max_faces = global_mesh.cell_neighbors.shape[1]
        cell_neighbors = -np.ones((num_local_cells, max_faces), dtype=int)
        for l_idx, g_idx in enumerate(l2g_cells):
            for i, neighbor_g in enumerate(global_mesh.cell_neighbors[g_idx]):
                if neighbor_g in g2l_cells:
                    cell_neighbors[l_idx, i] = g2l_cells[neighbor_g]

        local_mesh = cls(
            rank=rank,
            num_owned_cells=num_owned_cells,
            num_halo_cells=num_halo_cells,
            l2g_cells=l2g_cells,
            g2l_cells=g2l_cells,
            l2g_nodes=l2g_nodes,
            g2l_nodes=g2l_nodes,
            send_map=halo_info["send"],
            recv_map=halo_info["recv"],
        )

        local_mesh.node_coords = node_coords
        local_mesh.cell_connectivity = cell_connectivity
        local_mesh.cell_neighbors = cell_neighbors
        local_mesh.cell_type_ids = cell_type_ids
        local_mesh.cell_type_map = copy.deepcopy(global_mesh.cell_type_map)
        local_mesh.dimension = global_mesh.dimension

        local_mesh.num_cells = len(local_mesh.cell_connectivity)
        local_mesh.num_nodes = local_mesh.node_coords.shape[0]

        local_mesh.analyze_mesh()
        local_mesh._store_original_ordering()

        return local_mesh

    def _store_original_ordering(self):
        """Stores the initial state of the mesh before any reordering."""
        self._original_l2g_cells = self.l2g_cells.copy()
        self._original_g2l_cells = self.g2l_cells.copy()
        self._original_l2g_nodes = self.l2g_nodes.copy()
        self._original_g2l_nodes = self.g2l_nodes.copy()
        self._original_node_coords = self.node_coords.copy()
        self._original_cell_connectivity = copy.deepcopy(self.cell_connectivity)
        self._original_cell_neighbors = self.cell_neighbors.copy()
        if hasattr(self, "cell_type_ids"):
            self._original_cell_type_ids = self.cell_type_ids.copy()
        self._original_send_map = copy.deepcopy(self.send_map)
        self._original_recv_map = copy.deepcopy(self.recv_map)

    def _restore_original_cell_ordering(self):
        """Restores the mesh to its original cell ordering."""
        self.l2g_cells = self._original_l2g_cells.copy()
        self.g2l_cells = self._original_g2l_cells.copy()
        self.cell_connectivity = copy.deepcopy(self._original_cell_connectivity)
        self.cell_neighbors = self._original_cell_neighbors.copy()
        if hasattr(self, "_original_cell_type_ids"):
            self.cell_type_ids = self._original_cell_type_ids.copy()
        self.send_map = copy.deepcopy(self._original_send_map)
        self.recv_map = copy.deepcopy(self._original_recv_map)
        self.use_reordered_cells = False

    def _restore_original_node_ordering(self):
        """Restores the mesh to its original node ordering."""
        self.l2g_nodes = self._original_l2g_nodes.copy()
        self.g2l_nodes = self._original_g2l_nodes.copy()
        self.node_coords = self._original_node_coords.copy()
        # Connectivity must be restored to its state before node reordering
        self.cell_connectivity = copy.deepcopy(self._original_cell_connectivity)
        self.use_reordered_nodes = False

    def reorder_cells(self, strategy: str = "rcm", active: bool = True):
        """
        Reorders the cells of the mesh to improve locality.

        Args:
            strategy (str): The reordering strategy to use (e.g., 'rcm', 'bfs').
            active (bool): If False, restores the original cell ordering.
        """
        if not active:
            self._restore_original_cell_ordering()
        else:
            old_l2g = self.l2g_cells.copy()
            renumber_cells(self, strategy=strategy)
            # After renumbering, the g2l map is invalid and needs to be rebuilt
            self.g2l_cells = {g: l for l, g in enumerate(self.l2g_cells)}
            self.use_reordered_cells = True

            # Build a remap array: old_local_index -> new_local_index
            remap = np.array([self.g2l_cells[g] for g in old_l2g])

            # Update send_map and recv_map
            for rank, indices in self.send_map.items():
                self.send_map[rank] = [remap[i] for i in indices]
            for rank, indices in self.recv_map.items():
                self.recv_map[rank] = [remap[i] for i in indices]

        self.analyze_mesh()

    def reorder_nodes(self, strategy: str = "rcm", active: bool = True):
        """
        Reorders the nodes of the mesh to improve locality.

        Args:
            strategy (str): The reordering strategy to use.
            active (bool): If False, restores the original node ordering.
        """
        if not active:
            self._restore_original_node_ordering()
        else:
            renumber_nodes(self, strategy=strategy)
            # After renumbering, the g2l map is invalid and needs to be rebuilt
            self.g2l_nodes = {g: l for l, g in enumerate(self.l2g_nodes)}
            self.use_reordered_nodes = True
        self.analyze_mesh()


def create_local_meshes(
    global_mesh: CoreMesh,
    n_parts: Optional[int] = None,
    parts: Optional[np.ndarray] = None,
    partition_method: str = "metis",
    reorder_cells_strategy: Optional[str] = None,
    reorder_nodes_strategy: Optional[str] = None,
) -> List["LocalMesh"]:
    """
    Partitions a global mesh and creates a list of local mesh objects.

    This function orchestrates the mesh distribution process:
    1. Partitions the global mesh if `parts` are not provided.
    2. For each partition, constructs a `LocalMesh` object using the factory.
    3. Optionally reorders cells and/or nodes within each local mesh.

    Args:
        global_mesh: The complete, unpartitioned PolyMesh object.
        n_parts (Optional[int]): The desired number of partitions. Used if `parts` is not provided.
        parts (Optional[np.ndarray]): An array of partition indices for each cell. If provided, partitioning is skipped.
        partition_method (str): The algorithm for partitioning ('metis' or 'hierarchical').
        reorder_cells_strategy (Optional[str]): Cell reordering strategy ('rcm', 'bfs', etc.) or None to disable.
        reorder_nodes_strategy (Optional[str]): Node reordering strategy ('rcm', etc.) or None to disable.

    Returns:
        A list of LocalMesh objects, one for each partition.
    """
    if not global_mesh.cell_neighbors.size > 0:
        global_mesh.analyze_mesh()

    if parts is None:
        if n_parts is None or n_parts <= 0:
            raise ValueError(
                "Either n_parts > 0 or a valid parts array must be provided."
            )
        parts = partition_mesh(global_mesh, n_parts, method=partition_method)
        print_partition_summary(parts)
    else:
        # If parts are provided, derive n_parts from it, ignoring any passed n_parts
        n_parts = int(np.max(parts) + 1) if parts.size > 0 else 0

    halo_indices = _compute_halo_indices(global_mesh, parts)

    local_meshes = []
    if n_parts and n_parts > 0:
        for rank in range(n_parts):
            halo_info = halo_indices[rank]
            local_mesh = LocalMesh.from_global_mesh(global_mesh, parts, rank, halo_info)
            if reorder_cells_strategy:
                local_mesh.reorder_cells(strategy=reorder_cells_strategy)
            if reorder_nodes_strategy:
                local_mesh.reorder_nodes(strategy=reorder_nodes_strategy)
            local_meshes.append(local_mesh)

    return local_meshes
