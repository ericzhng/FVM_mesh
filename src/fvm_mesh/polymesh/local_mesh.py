# -*- coding: utf-8 -*-
"""
Tools for creating and managing local meshes in a distributed environment.

This module provides the `LocalMesh` class and related functions to manage
partitioned meshes for parallel computations. A `LocalMesh` represents the
portion of a global mesh assigned to a single process, including both the cells
owned by that process and the necessary halo cells from neighboring partitions.

Key Features:
- Creation of `LocalMesh` objects from a partitioned `CoreMesh`.
- Management of mappings between local and global indices for cells and nodes.
- Handling of halo cell information and communication maps (send/recv).
- In-place reordering of cells and nodes to optimize performance.

Classes:
    LocalMesh: Represents the mesh for a single partition in a distributed setup.

Functions:
    create_local_meshes: Partitions a global mesh and creates a list of local
        mesh objects.
"""

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
    """Determines which cells each partition needs to send to its neighbors."""
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
    """Builds communication maps for halo exchange between partitions."""
    n_parts = int(np.max(parts) + 1) if parts.size > 0 else 0
    if n_parts <= 1:
        return (
            {
                0: {
                    "owned_cells": list(range(mesh.num_cells)),
                    "halo_cells": [],
                    "send": {},
                    "recv": {},
                }
            }
            if mesh.num_cells > 0
            else {}
        )

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
            if cells_to_recv := global_send_map.get(neighbor_rank, {}).get(rank):
                halo_from_neighbors_g[neighbor_rank] = cells_to_recv
                halo_cells_g.extend(cells_to_recv)

        # The final local index for a halo cell is its position in the halo list
        # PLUS the number of owned cells.
        # e.g., if num_owned_cells is 100, the first halo cell has local index 100.
        g2l_halo_map = {g: l + len(owned_cells_g) for l, g in enumerate(halo_cells_g)}

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
    return [[g2l_map[g] for g in item_conn_g] for item_conn_g in global_conn]


class LocalMesh(PolyMesh):
    """
    Represents the mesh for a single partition in a distributed setup.

    This class holds the geometric and topological data for a subset of the
    global mesh, including cells owned by the current process and "halo" cells
    from neighboring processes.

    Attributes:
        rank (int): The rank of the partition this mesh belongs to.
        num_owned_cells (int): The number of cells owned by this partition.
        num_halo_cells (int): The number of halo cells.
        l2g_cells (np.ndarray): Map from local to global cell indices.
        g2l_cells (Dict[int, int]): Map from global to local cell indices.
        l2g_nodes (np.ndarray): Map from local to global node indices.
        g2l_nodes (Dict[int, int]): Map from global to local node indices.
        send_map (Dict[int, List[int]]): Halo communication map for sending data.
        recv_map (Dict[int, List[int]]): Halo communication map for receiving data.
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
    def from_global_mesh(
        cls, global_mesh: CoreMesh, parts: np.ndarray, rank: int, halo_info: Dict
    ) -> "LocalMesh":
        """
        Factory method to construct a LocalMesh for a specific partition.

        Args:
            global_mesh: The complete, unpartitioned mesh.
            parts: The result from the partitioning process.
            rank: The ID of the partition for which to create the local mesh.
            halo_info: A dictionary containing the halo information for this rank.
        """
        if not global_mesh.cell_neighbors.size > 0:
            global_mesh.analyze_mesh()

        l2g_cells, g2l_cells = cls._initialize_cell_maps(halo_info)
        l2g_nodes, g2l_nodes = cls._initialize_node_maps(global_mesh, l2g_cells)

        local_mesh = cls(
            rank=rank,
            num_owned_cells=len(halo_info["owned_cells"]),
            num_halo_cells=len(halo_info["halo_cells"]),
            l2g_cells=l2g_cells,
            g2l_cells=g2l_cells,
            l2g_nodes=l2g_nodes,
            g2l_nodes=g2l_nodes,
            send_map=halo_info["send"],
            recv_map=halo_info["recv"],
        )

        cls._build_local_mesh_data(local_mesh, global_mesh)
        local_mesh.analyze_mesh()
        local_mesh._store_original_ordering()
        return local_mesh

    @staticmethod
    def _initialize_cell_maps(halo_info: Dict) -> tuple[np.ndarray, Dict[int, int]]:
        """Initializes local-to-global and global-to-local cell maps."""
        l2g_cells = np.array(
            halo_info["owned_cells"] + halo_info["halo_cells"], dtype=int
        )
        g2l_cells = {g: l for l, g in enumerate(l2g_cells)}
        return l2g_cells, g2l_cells

    @staticmethod
    def _initialize_node_maps(
        global_mesh: CoreMesh, l2g_cells: np.ndarray
    ) -> tuple[np.ndarray, Dict[int, int]]:
        """Initializes local-to-global and global-to-local node maps."""
        local_cell_nodes_g = [global_mesh.cell_connectivity[g] for g in l2g_cells]
        unique_node_g_indices = np.unique(np.concatenate(local_cell_nodes_g))
        g2l_nodes = {g: l for l, g in enumerate(unique_node_g_indices)}
        return unique_node_g_indices, g2l_nodes

    @staticmethod
    def _build_local_mesh_data(local_mesh: "LocalMesh", global_mesh: CoreMesh) -> None:
        """Populates the LocalMesh with data from the global mesh."""
        local_mesh.node_coords = global_mesh.node_coords[local_mesh.l2g_nodes]
        local_mesh.cell_connectivity = _remap_connectivity(
            [global_mesh.cell_connectivity[g] for g in local_mesh.l2g_cells],
            local_mesh.g2l_nodes,
        )
        if global_mesh.cell_type_ids.size > 0:
            local_mesh.cell_type_ids = global_mesh.cell_type_ids[local_mesh.l2g_cells]

        num_local_cells = len(local_mesh.cell_connectivity)
        max_faces = global_mesh.cell_neighbors.shape[1]
        cell_neighbors = -np.ones((num_local_cells, max_faces), dtype=int)
        for l_idx, g_idx in enumerate(local_mesh.l2g_cells):
            for i, neighbor_g in enumerate(global_mesh.cell_neighbors[g_idx]):
                if neighbor_g in local_mesh.g2l_cells:
                    cell_neighbors[l_idx, i] = local_mesh.g2l_cells[neighbor_g]
        local_mesh.cell_neighbors = cell_neighbors

        local_mesh.cell_type_map = copy.deepcopy(global_mesh.cell_type_map)
        local_mesh.dimension = global_mesh.dimension
        local_mesh.num_cells = len(local_mesh.cell_connectivity)
        local_mesh.num_nodes = local_mesh.node_coords.shape[0]

    def _store_original_ordering(self) -> None:
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

    def _restore_original_cell_ordering(self) -> None:
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

    def _restore_original_node_ordering(self) -> None:
        """Restores the mesh to its original node ordering."""
        self.l2g_nodes = self._original_l2g_nodes.copy()
        self.g2l_nodes = self._original_g2l_nodes.copy()
        self.node_coords = self._original_node_coords.copy()
        self.cell_connectivity = copy.deepcopy(self._original_cell_connectivity)
        self.use_reordered_nodes = False

    def reorder_cells(self, strategy: str = "rcm", active: bool = True) -> None:
        """
        Reorders the cells of the mesh to improve locality.

        Args:
            strategy: The reordering strategy to use (e.g., 'rcm', 'bfs').
            active: If False, restores the original cell ordering.
        """
        if not active:
            self._restore_original_cell_ordering()
            self.analyze_mesh()
            return

        old_l2g = self.l2g_cells.copy()
        renumber_cells(self, strategy=strategy)
        self.g2l_cells = {g: l for l, g in enumerate(self.l2g_cells)}
        self.use_reordered_cells = True

        remap = np.array([self.g2l_cells[g] for g in old_l2g])
        for rank, indices in self.send_map.items():
            self.send_map[rank] = [remap[i] for i in indices]
        for rank, indices in self.recv_map.items():
            self.recv_map[rank] = [remap[i] for i in indices]

        self.analyze_mesh()

    def reorder_nodes(self, strategy: str = "rcm", active: bool = True) -> None:
        """
        Reorders the nodes of the mesh to improve locality.

        Args:
            strategy: The reordering strategy to use.
            active: If False, restores the original node ordering.
        """
        if not active:
            self._restore_original_node_ordering()
        else:
            renumber_nodes(self, strategy=strategy)
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

    Args:
        global_mesh: The complete, unpartitioned PolyMesh object.
        n_parts: The desired number of partitions.
        parts: An array of partition indices for each cell.
        partition_method: The algorithm for partitioning ('metis' or 'hierarchical').
        reorder_cells_strategy: Cell reordering strategy ('rcm', 'bfs', etc.).
        reorder_nodes_strategy: Node reordering strategy ('rcm', etc.).

    Returns:
        A list of LocalMesh objects, one for each partition.
    """
    if not global_mesh.cell_neighbors.size > 0:
        global_mesh.analyze_mesh()

    if parts is None:
        if n_parts is None or n_parts <= 0:
            raise ValueError("n_parts > 0 or a valid parts array must be provided.")
        parts = partition_mesh(global_mesh, n_parts, method=partition_method)
        print_partition_summary(parts)
    else:
        n_parts = int(np.max(parts) + 1) if parts.size > 0 else 0

    halo_indices = _compute_halo_indices(global_mesh, parts)
    local_meshes = []
    if n_parts and n_parts > 0:
        for rank in range(n_parts):
            if rank not in halo_indices:
                continue
            local_mesh = LocalMesh.from_global_mesh(
                global_mesh, parts, rank, halo_indices[rank]
            )
            if reorder_cells_strategy:
                local_mesh.reorder_cells(strategy=reorder_cells_strategy)
            if reorder_nodes_strategy:
                local_mesh.reorder_nodes(strategy=reorder_nodes_strategy)
            local_meshes.append(local_mesh)

    return local_meshes
