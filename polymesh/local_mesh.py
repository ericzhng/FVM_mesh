from typing import Dict, List, Tuple

import numpy as np

from polymesh.mesh import Mesh
from polymesh.partition import PartitionResult


class LocalMesh:
    """
    Represents the mesh for a single partition (process) in a distributed mesh setup.

    This class holds the subset of the mesh relevant to one partition, including
    both the cells it owns and the halo cells it needs for communication. All
    entities (nodes, faces, cells) are renumbered into a local, contiguous,
    0-based index space to allow for efficient computation.

    Attributes:
        rank (int): The rank of the partition this mesh belongs to.
        mesh (Mesh): A new Mesh object containing only the local entities.
        g2l_cells (Dict[int, int]): Mapping from global cell ID to local cell ID.
        l2g_cells (np.ndarray): Mapping from local cell ID to global cell ID.
        g2l_nodes (Dict[int, int]): Mapping from global node ID to local node ID.
        l2g_nodes (np.ndarray): Mapping from local node ID to global node ID.
        num_owned_cells (int): The number of cells owned by this partition.
        num_halo_cells (int): The number of halo cells in this partition.
        send_map (Dict[int, List[int]]): Communication map for sending data.
        recv_map (Dict[int, List[int]]): Communication map for receiving data.
    """

    def __init__(self, global_mesh: Mesh, part_result: PartitionResult, rank: int):
        """
        Constructs a LocalMesh for a specific partition.

        Args:
            global_mesh: The complete, unpartitioned mesh.
            part_result: The result from the partitioning process.
            rank: The ID of the partition for which to create the local mesh.
        """
        self.rank = rank
        if not global_mesh._is_analyzed:
            global_mesh.analyze_mesh()

        halo_info = part_result.halo_indices[rank]
        owned_cells_g = halo_info["owned_cells"]
        halo_cells_g = halo_info["halo_cells"]

        self.num_owned_cells = len(owned_cells_g)
        self.num_halo_cells = len(halo_cells_g)

        # 1. Create local numbering for cells (owned first, then halo)
        self.l2g_cells = np.array(owned_cells_g + halo_cells_g, dtype=int)
        self.g2l_cells = {g: l for l, g in enumerate(self.l2g_cells)}

        # 2. Identify all unique nodes required for the local cells
        local_cell_nodes_g = global_mesh.cell_nodes[self.l2g_cells]
        unique_node_g_indices = np.unique(np.concatenate(local_cell_nodes_g))
        self.l2g_nodes = unique_node_g_indices
        self.g2l_nodes = {g: l for l, g in enumerate(self.l2g_nodes)}

        # 3. Create new, locally indexed mesh data
        local_nodes = global_mesh.nodes[self.l2g_nodes]
        local_cell_nodes = self._remap_connectivity(local_cell_nodes_g, self.g2l_nodes)

        # 4. Create the new local Mesh object
        self.mesh = Mesh(nodes=local_nodes, cells=local_cell_nodes)
        self.mesh.analyze_mesh()  # Analyze to build faces, neighbors, etc.

        # 5. Remap cell neighbors to local indices
        self._remap_cell_neighbors(global_mesh)

        # 6. Store communication maps
        self.send_map = halo_info["send"]
        self.recv_map = halo_info["recv"]

    def _remap_connectivity(
        self, global_conn: List[np.ndarray], g2l_map: Dict[int, int]
    ) -> List[np.ndarray]:
        """Remaps a global connectivity list to a local one."""
        local_conn = []
        for item_conn_g in global_conn:
            local_conn.append(np.array([g2l_map[g] for g in item_conn_g], dtype=int))
        return local_conn

    def _remap_cell_neighbors(self, global_mesh: Mesh):
        """Remaps the cell_neighbors array to use local cell indices."""
        # This is a more complex operation as neighbors might not exist in the
        # local mesh (i.e., they are external boundaries).
        new_cell_neighbors = []
        for l_idx, g_idx in enumerate(self.l2g_cells):

            # Find the corresponding face in the local mesh
            # This is a simplification; a robust method would map faces properly

            local_neighbors = []
            for neighbor_g in global_mesh.cell_neighbors[g_idx]:
                if neighbor_g in self.g2l_cells:
                    local_neighbors.append(self.g2l_cells[neighbor_g])
                else:
                    local_neighbors.append(-1)  # Boundary face

            # This simplistic remapping might not match face order.
            # For a real solver, a full face remapping is needed.
            # Here, we just re-populate the neighbor list based on global data.
            self.mesh.cell_neighbors[l_idx] = np.array(local_neighbors, dtype=int)
