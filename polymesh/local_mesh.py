import copy
from typing import Dict, List, Set, Sequence

import numpy as np

from polymesh.reorder import renumber_cells

from .poly_mesh import PolyMesh
from .core_mesh import CoreMesh
from .partition import partition_mesh, print_partition_summary


def _find_send_candidates(
    mesh: CoreMesh, parts: np.ndarray
) -> Dict[int, Dict[int, List[int]]]:
    """
    Determines which cells each partition needs to send to its neighbors.

    This is the first step in building the halo communication maps. It iterates
    through all cell-neighbor pairs in the global mesh. If a pair crosses a
    partition boundary, the owner cell is marked as a "send candidate" to the
    neighbor's partition.

    Args:
        mesh: The global mesh object.
        parts: An array mapping each global cell ID to its partition ID (rank).

    Returns:
        A dictionary where keys are partition ranks. Each value is another
        dictionary mapping a neighbor rank to a sorted list of global cell IDs
        to be sent from the key rank to the neighbor rank.
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

    # Sort the sets for deterministic ordering
    global_send_map: Dict[int, Dict[int, List[int]]] = {r: {} for r in range(n_parts)}
    for rank, send_dict in send_candidates.items():
        for neighbor_rank, g_indices_set in send_dict.items():
            global_send_map[rank][neighbor_rank] = sorted(list(g_indices_set))

    return global_send_map


def _compute_halo_indices(mesh: CoreMesh, parts: np.ndarray) -> Dict:
    """
    Builds communication maps for halo exchange between partitions.

    This method creates a set of mappings for each partition (rank) that
    enables efficient, index-based halo data exchange. It avoids costly
    searches by pre-calculating the local indices for sending owned cells
    and receiving halo cells.

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

    # For each partition, build the final local index-based maps.
    out: Dict[int, Dict] = {}
    for rank in range(n_parts):
        # 1. Identify owned and halo cells for the current rank
        owned_cells_g = np.where(parts == rank)[0]
        g2l_owned_map = {g: l for l, g in enumerate(owned_cells_g)}

        halo_cells_g: List[int] = []
        halo_from_neighbors_g: Dict[int, List[int]] = {}

        # Collect all neighbors that this rank interacts with (sends to or receives from)
        all_neighbors = set(global_send_map.get(rank, {}).keys())
        for r, send_map in global_send_map.items():
            if rank in send_map:
                all_neighbors.add(r)

        # Determine which cells this rank needs to receive from its neighbors
        for neighbor_rank in sorted(list(all_neighbors)):
            cells_to_recv = global_send_map.get(neighbor_rank, {}).get(rank)
            if cells_to_recv:
                halo_from_neighbors_g[neighbor_rank] = cells_to_recv
                halo_cells_g.extend(cells_to_recv)

        g2l_halo_map = {g: l for l, g in enumerate(halo_cells_g)}

        # 2. Create local index maps for sending and receiving
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


class LocalMesh(PolyMesh):
    """
    Represents the mesh for a single partition in a distributed setup.

    This class holds the subset of the mesh relevant to one partition, including
    both the cells it owns and the halo cells it needs for communication. All
    entities (nodes, faces, cells) are renumbered into a local, contiguous,
    0-based index space to allow for efficient computation.

    Attributes:
        rank (int): The rank of the partition this mesh belongs to.
        g2l_cells (Dict[int, int]): Mapping from global cell ID to local cell ID.
        l2g_cells (np.ndarray): Mapping from local cell ID to global cell ID.
        g2l_nodes (Dict[int, int]): Mapping from global node ID to local node ID.
        l2g_nodes (np.ndarray): Mapping from local node ID to global node ID.
        num_owned_cells (int): The number of cells owned by this partition.
        num_halo_cells (int): The number of halo cells in this partition.
        send_map (Dict[int, List[int]]): Communication map for sending data.
        recv_map (Dict[int, List[int]]): Communication map for receiving data.
    """

    def __init__(self, global_mesh: CoreMesh, parts: np.ndarray, rank: int):
        """
        Constructs a LocalMesh for a specific partition.

        Args:
            global_mesh: The complete, unpartitioned mesh.
            parts: The result from the partitioning process.
            rank: The ID of the partition for which to create the local mesh.
        """
        super().__init__()
        self.rank = rank
        if not global_mesh.cell_neighbors.size > 0:
            global_mesh._extract_neighbors()

        # 1. Determine halo and communication patterns from the global mesh
        halo_indices = _compute_halo_indices(global_mesh, parts)
        halo_info = halo_indices[rank]

        # 2. Set up local cell mappings (owned first, then halo)
        self._initialize_cell_maps(halo_info)

        # 3. Identify required nodes and set up local node mappings
        self._initialize_node_maps(global_mesh)

        # 4. Build the local mesh structure (nodes and connectivity)
        self._build_local_mesh_data(global_mesh)

        # 5. Analyze the newly created local mesh to build faces, volumes, etc.
        self.analyze_mesh()

        # 6. Remap cell neighbors to use local indices
        self._remap_neighbors(global_mesh)

        # 7. Store communication maps for halo exchange
        self.send_map = halo_info["send"]
        self.recv_map = halo_info["recv"]

    def _initialize_cell_maps(self, halo_info: Dict):
        """Initializes local-global cell mappings."""
        owned_cells_g = halo_info["owned_cells"]
        halo_cells_g = halo_info["halo_cells"]
        self.num_owned_cells = len(owned_cells_g)
        self.num_halo_cells = len(halo_cells_g)
        # local to global cell mapping
        self.l2g_cells = np.array(owned_cells_g + halo_cells_g, dtype=int)
        # global to local cell mapping
        self.g2l_cells = {g: l for l, g in enumerate(self.l2g_cells)}

    def _initialize_node_maps(self, global_mesh: CoreMesh):
        """Identifies required nodes and creates local-global node mappings."""
        local_cell_nodes_g = [global_mesh.cell_connectivity[g] for g in self.l2g_cells]
        unique_node_g_indices = np.unique(np.concatenate(local_cell_nodes_g))
        self.l2g_nodes = unique_node_g_indices
        self.g2l_nodes = {g: l for l, g in enumerate(self.l2g_nodes)}

    def _build_local_mesh_data(self, global_mesh: CoreMesh):
        """Creates the local node coordinates and cell connectivity."""
        self.node_coords = global_mesh.node_coords[self.l2g_nodes]
        local_cell_nodes_g = [global_mesh.cell_connectivity[g] for g in self.l2g_cells]
        self.cell_connectivity = self._remap_connectivity(
            local_cell_nodes_g, self.g2l_nodes
        )
        self.dimension = global_mesh.dimension
        self.cell_type_map = copy.deepcopy(global_mesh.cell_type_map)
        if global_mesh.cell_type_ids.size > 0:
            self.cell_type_ids = global_mesh.cell_type_ids[self.l2g_cells]
        else:
            self.cell_type_ids = np.array([])

        self.num_cells = len(self.cell_connectivity)
        self.num_nodes = self.node_coords.shape[0]

    def _remap_connectivity(
        self, global_conn: Sequence[np.ndarray], g2l_map: Dict[int, int]
    ) -> List[List[int]]:
        """Remaps a global connectivity sequence to a local one."""
        local_conn = []
        for item_conn_g in global_conn:
            local_conn.append([g2l_map[g] for g in item_conn_g])
        return local_conn

    def _remap_neighbors(self, global_mesh: CoreMesh):
        """Remaps the cell_neighbors array to use local cell indices."""
        # This assumes self.cell_neighbors has been allocated by analyze_mesh()
        for l_idx, g_idx in enumerate(self.l2g_cells):
            local_neighbors = []
            if g_idx < len(global_mesh.cell_neighbors):
                for neighbor_g in global_mesh.cell_neighbors[g_idx]:
                    if neighbor_g in self.g2l_cells:
                        local_neighbors.append(self.g2l_cells[neighbor_g])
                    else:
                        local_neighbors.append(-1)  # Boundary face

            # Ensure the row in cell_neighbors is correctly sized and assigned
            if l_idx < self.cell_neighbors.shape[0]:
                # Pad with -1 if necessary
                padded_neighbors = np.full(self.cell_neighbors.shape[1], -1, dtype=int)
                padded_neighbors[: len(local_neighbors)] = local_neighbors
                self.cell_neighbors[l_idx] = padded_neighbors


def create_local_meshes(
    global_mesh: CoreMesh, n_parts: int, partition_method: str = "metis"
) -> List["LocalMesh"]:
    """
    Partitions a global mesh and creates a list of local mesh objects.

    This function orchestrates the mesh distribution process:
    1. Partitions the global mesh into the specified number of parts.
    2. For each partition, constructs a `LocalMesh` object containing
       only the data relevant to that partition (owned cells, halo cells,
       and local connectivity).

    Args:
        global_mesh: The complete, unpartitioned PolyMesh object.
        n_parts: The desired number of partitions.
        partition_method: The algorithm to use for partitioning ('metis' or 'hierarchical').

    Returns:
        A list of LocalMesh objects, one for each partition.
    """
    if not global_mesh.cell_neighbors.size > 0:
        global_mesh._extract_neighbors()

    # 1. Partition the global mesh
    parts = partition_mesh(global_mesh, n_parts, method=partition_method)
    print_partition_summary(parts)

    # 2. Create a LocalMesh for each partition
    local_meshes = []
    if n_parts > 0:
        for rank in range(n_parts):
            local_mesh = LocalMesh(global_mesh, parts, rank)
            renumber_cells(local_mesh, strategy="bfs")
            local_meshes.append(local_mesh)

    return local_meshes
