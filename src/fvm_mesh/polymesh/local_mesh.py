# -*- coding: utf-8 -*-
"""
Tools for creating and managing local meshes in a distributed environment.

This module provides the `LocalMesh` class to manage partitioned meshes for
parallel computations. A `LocalMesh` represents the portion of a global mesh
assigned to a single process, including both the cells owned by that process
and the necessary halo cells from neighboring partitions.

Key Features:
- Creation of `LocalMesh` objects from a partitioned `PolyMesh`.
- Management of mappings between local and global indices for cells and nodes.
- Handling of halo cell information and communication maps (send/recv).
- In-place reordering of cells and nodes to optimize performance.

Classes:
    LocalMesh: Represents the mesh for a single partition in a distributed setup.
"""

import copy
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from .poly_mesh import PolyMesh
from .reorder import renumber_cells, renumber_nodes


def _remap_connectivity(
    global_connectivity: Sequence[npt.NDArray[np.int_]], g2l_map: Dict[int, int]
) -> List[List[int]]:
    """
    Remaps a global connectivity sequence to a local one.

    Args:
        global_connectivity: A sequence of NumPy arrays, where each array contains
            global indices (e.g., global node indices for a cell).
        g2l_map: A dictionary mapping global indices to local indices.

    Returns:
        A list of lists, where each inner list contains the remapped local indices.

    Raises:
        KeyError: If a global index in the connectivity data is not found in the
                  global-to-local map.
    """
    try:
        return [
            [g2l_map[g] for g in item_conn_g] for item_conn_g in global_connectivity
        ]
    except KeyError as e:
        raise KeyError(f"Global index {e} not found in the global-to-local map.") from e


def _initialize_cell_maps(
    halo_info: Dict[str, List[int]],
) -> Tuple[npt.NDArray[np.int_], Dict[int, int]]:
    """
    Initializes local-to-global and global-to-local cell maps.

    Args:
        halo_info: A dictionary containing "owned_cells" and "halo_cells"
                   lists of global cell indices.

    Returns:
        A tuple containing:
        - l2g_cells: An array mapping local cell indices to global cell indices.
        - g2l_cells: A dictionary mapping global cell indices to local cell indices.
    """
    if not halo_info.get("owned_cells"):
        raise ValueError("halo_info must contain a non-empty 'owned_cells' list.")

    l2g_cells = np.array(
        halo_info["owned_cells"] + halo_info.get("halo_cells", []), dtype=int
    )
    g2l_cells = {g: l for l, g in enumerate(l2g_cells)}
    return l2g_cells, g2l_cells


def _initialize_node_maps(
    global_mesh: PolyMesh, l2g_cells: npt.NDArray[np.int_]
) -> Tuple[npt.NDArray[np.int_], Dict[int, int]]:
    """
    Initializes local-to-global and global-to-local node maps.

    This function identifies all unique nodes required for the local cells
    and creates the necessary mappings.

    Args:
        global_mesh: The complete, unpartitioned mesh.
        l2g_cells: An array mapping local cell indices to global cell indices.

    Returns:
        A tuple containing:
        - l2g_nodes: An array mapping local node indices to global node indices.
        - g2l_nodes: A dictionary mapping global node indices to local node indices.
    """
    if l2g_cells.size == 0:
        return np.array([], dtype=int), {}

    local_cell_nodes_g = [global_mesh.cell_node_connectivity[g] for g in l2g_cells]
    unique_node_g_indices = np.unique(np.concatenate(local_cell_nodes_g))
    g2l_nodes = {g: l for l, g in enumerate(unique_node_g_indices)}
    return unique_node_g_indices, g2l_nodes


class LocalMesh(PolyMesh):
    """
    Represents the mesh for a single partition in a distributed setup.

    This class holds the geometric and topological data for a subset of the
    global mesh, including cells owned by the current process and "halo" cells
    from neighboring processes required for communication.

    Attributes:
        rank (int): The rank of the partition this mesh belongs to.
        num_owned_cells (int): The number of cells owned by this partition.
        num_halo_cells (int): The number of halo cells.
        l2g_cells (npt.NDArray[np.int_]): Map from local to global cell indices.
        g2l_cells (Dict[int, int]): Map from global to local cell indices.
        l2g_nodes (npt.NDArray[np.int_]): Map from local to global node indices.
        g2l_nodes (Dict[int, int]): Map from global to local node indices.
        send_map (Dict[int, List[int]]): Halo communication map for sending data.
        recv_map (Dict[int, List[int]]): Halo communication map for receiving data.
        use_reordered_cells (bool): Flag indicating if cells are currently reordered.
        use_reordered_nodes (bool): Flag indicating if nodes are currently reordered.
    """

    def __init__(
        self,
        rank: int,
        num_owned_cells: int,
        num_halo_cells: int,
        l2g_cells: npt.NDArray[np.int_],
        g2l_cells: Dict[int, int],
        l2g_nodes: npt.NDArray[np.int_],
        g2l_nodes: Dict[int, int],
        send_map: Dict[int, List[int]],
        recv_map: Dict[int, List[int]],
    ):
        """
        Initializes a LocalMesh object.

        Args:
            rank: The rank of the partition this mesh belongs to.
            num_owned_cells: The number of cells owned by this partition.
            num_halo_cells: The number of halo cells.
            l2g_cells: Array mapping local cell indices to global cell indices.
            g2l_cells: Dictionary mapping global cell indices to local cell indices.
            l2g_nodes: Array mapping local node indices to global node indices.
            g2l_nodes: Dictionary mapping global node indices to local node indices.
            send_map: Map of {neighbor_rank: [local_cell_indices_to_send]}.
            recv_map: Map of {neighbor_rank: [local_cell_indices_to_receive]}.
        """
        super().__init__()
        if rank < 0:
            raise ValueError("Rank must be a non-negative integer.")
        if num_owned_cells <= 0:
            raise ValueError("Number of owned cells must be positive.")

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

    def _populate_boundary_faces(self, global_mesh: PolyMesh) -> None:
        """
        Populates the local mesh with boundary face information.

        This method identifies boundary faces from the global mesh that are
        entirely contained within the current local mesh partition.

        Args:
            global_mesh: The global PolyMesh object.
        """
        local_boundary_face_nodes: List[List[int]] = []
        local_boundary_face_tags: List[int] = []

        if getattr(global_mesh, "boundary_face_nodes", np.array([])).size > 0:
            for i, face_nodes_g in enumerate(global_mesh.boundary_face_nodes):
                if all(node_g in self.g2l_nodes for node_g in face_nodes_g):
                    face_nodes_l = [self.g2l_nodes[node_g] for node_g in face_nodes_g]
                    local_boundary_face_nodes.append(face_nodes_l)
                    local_boundary_face_tags.append(global_mesh.boundary_face_tags[i])

        # Using dtype=object because boundary faces can have varying numbers of nodes.
        # This creates a NumPy array of lists, which may have performance implications.
        self.boundary_face_nodes = (
            np.array(local_boundary_face_nodes, dtype=object)
            if local_boundary_face_nodes
            else np.array([], dtype=int)
        )
        self.boundary_face_tags = np.array(local_boundary_face_tags, dtype=int)
        self.boundary_patch_map = copy.deepcopy(global_mesh.boundary_patch_map)

    def _build_poly_mesh_data(self, global_mesh: PolyMesh) -> None:
        """
        Populates the LocalMesh with geometric and topological data.

        Args:
            global_mesh: The global PolyMesh object from which to draw data.
        """
        self.node_coords = global_mesh.node_coords[self.l2g_nodes]
        self.cell_node_connectivity = _remap_connectivity(
            [global_mesh.cell_node_connectivity[g] for g in self.l2g_cells],
            self.g2l_nodes,
        )

        if global_mesh.cell_element_types.size > 0:
            self.cell_element_types = global_mesh.cell_element_types[self.l2g_cells]

        self.element_type_properties = copy.deepcopy(
            getattr(global_mesh, "element_type_properties", {})
        )
        self.dimension = global_mesh.dimension
        self.n_cells = len(self.cell_node_connectivity)
        self.n_nodes = self.node_coords.shape[0]

        self._populate_boundary_faces(global_mesh)

    @classmethod
    def from_global_mesh(
        cls,
        global_mesh: PolyMesh,
        halo_info: Dict[str, Any],
        rank: int,
        store_order: bool = False,
    ) -> "LocalMesh":
        """
        Factory method to construct a LocalMesh for a specific partition.

        Args:
            global_mesh: The complete, unpartitioned mesh.
            halo_info: A dictionary with halo information for this rank.
            rank: The ID of the partition for which to create the local mesh.
            store_order: If True, stores the original cell and node ordering, allowing
                    for later restoration via `reorder_cells(restore=True)` or
                    `reorder_nodes(restore=True)`. Defaults to False.

        Returns:
            A new LocalMesh instance for the specified rank.

        Raises:
            ValueError: If the global mesh has not been analyzed or if halo
                        information is incomplete.
        """
        if getattr(global_mesh, "cell_neighbors", np.array([])).size == 0:
            raise ValueError("Mesh must be analyzed before creating a LocalMesh.")

        l2g_cells, g2l_cells = _initialize_cell_maps(halo_info)
        l2g_nodes, g2l_nodes = _initialize_node_maps(global_mesh, l2g_cells)

        local_mesh = cls(
            rank=rank,
            num_owned_cells=len(halo_info["owned_cells"]),
            num_halo_cells=len(halo_info.get("halo_cells", [])),
            l2g_cells=l2g_cells,
            g2l_cells=g2l_cells,
            l2g_nodes=l2g_nodes,
            g2l_nodes=g2l_nodes,
            send_map=halo_info.get("send", {}),
            recv_map=halo_info.get("recv", {}),
        )

        local_mesh._build_poly_mesh_data(global_mesh)

        if store_order:
            local_mesh._store_original_ordering()

        return local_mesh

    def plot(
        self,
        filepath: str = "mesh_plot.png",
        show_cells: bool = True,
        show_nodes: bool = True,
    ) -> None:
        parts = np.zeros(self.n_cells, dtype=int)
        parts[self.num_owned_cells :] = 1
        super().plot(filepath, parts, show_cells, show_nodes)

    def reorder_cells(self, strategy: str = "rcm", restore: bool = False) -> None:
        """
        Reorders the cells of the mesh to improve locality.

        This is a stateful operation that modifies the mesh in-place.
        If 'restore' is True, the original cell ordering is restored.

        Args:
            strategy: The reordering strategy to use (e.g., 'rcm', 'bfs').
            restore: If True, restores the original ordering. Defaults to False.
        """
        if restore:
            self._restore_original_cell_ordering()
            self.analyze_mesh()
            return

        old_l2g_cells = self.l2g_cells.copy()
        renumber_cells(self, strategy=strategy)
        self.g2l_cells = {g: l for l, g in enumerate(self.l2g_cells)}
        self.use_reordered_cells = True

        # Remap send/recv maps based on new local cell indices
        remap_old_l_to_new_l = {
            old_l: self.g2l_cells[g] for old_l, g in enumerate(old_l2g_cells)
        }
        self.send_map = {
            rank: [remap_old_l_to_new_l[i] for i in indices]
            for rank, indices in self.send_map.items()
        }
        self.recv_map = {
            rank: [remap_old_l_to_new_l[i] for i in indices]
            for rank, indices in self.recv_map.items()
        }
        self.analyze_mesh()

    def reorder_nodes(self, strategy: str = "rcm", restore: bool = False) -> None:
        """
        Reorders the nodes of the mesh to improve locality.

        This is a stateful operation that modifies the mesh in-place.
        If 'restore' is True, the original node ordering is restored.

        Args:
            strategy: The reordering strategy to use (e.g., 'rcm').
            restore: If True, restores the original ordering. Defaults to False.
        """
        if restore:
            self._restore_original_node_ordering()
            self.analyze_mesh()
            return

        renumber_nodes(self, strategy=strategy)
        self.g2l_nodes = {g: l for l, g in enumerate(self.l2g_nodes)}
        self.use_reordered_nodes = True
        self.analyze_mesh()

    def _store_original_ordering(self) -> None:
        """Stores the initial state of the mesh for later restoration."""
        self._original_l2g_cells = self.l2g_cells.copy()
        self._original_g2l_cells = self.g2l_cells.copy()
        self._original_l2g_nodes = self.l2g_nodes.copy()
        self._original_g2l_nodes = self.g2l_nodes.copy()
        self._original_node_coords = self.node_coords.copy()
        self._original_cell_node_connectivity = [
            list(conn) for conn in self.cell_node_connectivity
        ]
        self._original_cell_element_types = self.cell_element_types.copy()
        self._original_send_map = copy.deepcopy(self.send_map)
        self._original_recv_map = copy.deepcopy(self.recv_map)

    def _restore_original_cell_ordering(self) -> None:
        """Restores the mesh to its original cell ordering."""
        if not hasattr(self, "_original_l2g_cells"):
            return  # Nothing to restore

        self.l2g_cells = self._original_l2g_cells.copy()
        self.g2l_cells = self._original_g2l_cells.copy()
        self.cell_node_connectivity = [
            list(conn) for conn in self._original_cell_node_connectivity
        ]
        self.cell_element_types = self._original_cell_element_types.copy()
        self.send_map = copy.deepcopy(self._original_send_map)
        self.recv_map = copy.deepcopy(self._original_recv_map)
        self.use_reordered_cells = False

    def _restore_original_node_ordering(self) -> None:
        """Restores the mesh to its original node ordering."""
        if not hasattr(self, "_original_l2g_nodes"):
            return  # Nothing to restore

        self.l2g_nodes = self._original_l2g_nodes.copy()
        self.g2l_nodes = self._original_g2l_nodes.copy()
        self.node_coords = self._original_node_coords.copy()
        self.cell_node_connectivity = [
            list(conn) for conn in self._original_cell_node_connectivity
        ]
        self.use_reordered_nodes = False
