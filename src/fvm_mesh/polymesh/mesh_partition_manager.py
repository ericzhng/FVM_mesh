# -*- coding: utf-8 -*-
"""
A manager for partitioning meshes and creating local mesh instances.

This module provides the `MeshPartitionManager` class, which handles the
logic of taking a global `PolyMesh`, partitioning it, computing halo cell
information, and creating multiple `LocalMesh` instances for distributed
environments.
"""

from typing import Any, Dict, List, Set, Optional

import numpy as np
import numpy.typing as npt

from .poly_mesh import PolyMesh
from .local_mesh import LocalMesh
from .partition import partition_mesh, print_partition_summary


class MeshPartitionManager:
    """
    Manages the partitioning of a global mesh and creation of local meshes.
    This class is designed as a stateless manager, providing class methods
    to perform partitioning and mesh creation tasks.
    """

    @staticmethod
    def _find_send_candidates(
        global_mesh: PolyMesh, cell_partitions: npt.NDArray[np.int_]
    ) -> Dict[int, Dict[int, List[int]]]:
        """
        Determines which cells each partition needs to send to its neighbors.

        This method identifies cells adjacent to cells in a different partition,
        which are candidates for halo exchange.

        Args:
            global_mesh: The global PolyMesh object.
            cell_partitions: An array where `cell_partitions[i]` is the
                             partition ID of global cell `i`.

        Returns:
            A map where keys are sender partition IDs and values are another
            map of {receiver_partition_ID: [global_cell_indices_to_send]}.
        """
        if cell_partitions.size == 0:
            return {}

        n_parts = int(np.max(cell_partitions) + 1)
        send_candidates: Dict[int, Dict[int, Set[int]]] = {
            r: {} for r in range(n_parts)
        }

        for g_idx, neighbors in enumerate(global_mesh.cell_neighbors):
            owner_part = cell_partitions[g_idx]
            for neighbor_g_idx in neighbors:
                if neighbor_g_idx != -1:
                    neighbor_part = cell_partitions[neighbor_g_idx]
                    if owner_part != neighbor_part:
                        send_candidates[owner_part].setdefault(
                            int(neighbor_part), set()
                        ).add(g_idx)

        # Convert sets to sorted lists for deterministic behavior
        global_send_map: Dict[int, Dict[int, List[int]]] = {
            r: {} for r in range(n_parts)
        }
        for rank, send_dict in send_candidates.items():
            for neighbor_rank, g_indices_set in send_dict.items():
                global_send_map[rank][neighbor_rank] = sorted(list(g_indices_set))

        return global_send_map

    @staticmethod
    def _compute_halo_indices(
        global_mesh: PolyMesh,
        cell_partitions: npt.NDArray[np.int_],
        global_send_map: Dict[int, Dict[int, List[int]]],
    ) -> Dict[int, Dict[str, Any]]:
        """
        Builds communication maps for halo exchange between partitions.

        This function determines for each partition:
        1. Its owned cells.
        2. Its halo cells (received from neighbors).
        3. A "send map" of owned cells to send to each neighbor.
        4. A "recv map" of halo cells to receive from each neighbor.

        Args:
            global_mesh: The global PolyMesh object.
            cell_partitions: An array mapping global cell indices to partition IDs.
            global_send_map: A map of cells to be sent between partitions.

        Returns:
            A dictionary where keys are partition IDs. Each value is a dictionary
            containing "owned_cells", "halo_cells", "send", and "recv" maps.
        """
        if cell_partitions.size == 0:
            return {}

        n_parts = int(np.max(cell_partitions) + 1)
        if n_parts <= 1:
            return (
                {
                    0: {
                        "owned_cells": list(range(global_mesh.n_cells)),
                        "halo_cells": [],
                        "send": {},
                        "recv": {},
                    }
                }
                if global_mesh.n_cells > 0
                else {}
            )

        halo_data: Dict[int, Dict[str, Any]] = {}
        for rank in range(n_parts):
            owned_cells_g = np.where(cell_partitions == rank)[0]
            g2l_owned_map = {g: l for l, g in enumerate(owned_cells_g)}

            # Collect all halo cells this rank will receive
            halo_cells_g: List[int] = []
            halo_from_neighbors_g: Dict[int, List[int]] = {}
            for sender_rank, send_dict in global_send_map.items():
                if rank in send_dict:
                    cells_to_recv = send_dict[rank]
                    halo_from_neighbors_g[sender_rank] = cells_to_recv
                    halo_cells_g.extend(cells_to_recv)

            # Create a global-to-local map for halo cells. Local indices for
            # halo cells start after the owned cells.
            g2l_halo_map = {
                g: l + len(owned_cells_g) for l, g in enumerate(halo_cells_g)
            }

            # Create local send map from the global send map
            send_map_local: Dict[int, List[int]] = {}
            if rank in global_send_map:
                for neighbor_rank, g_indices_to_send in global_send_map[rank].items():
                    send_map_local[neighbor_rank] = [
                        g2l_owned_map[g] for g in g_indices_to_send
                    ]

            # Create local recv map
            recv_map_local: Dict[int, List[int]] = {}
            for neighbor_rank, g_indices_to_recv in halo_from_neighbors_g.items():
                recv_map_local[neighbor_rank] = [
                    g2l_halo_map[g] for g in g_indices_to_recv
                ]

            halo_data[rank] = {
                "owned_cells": owned_cells_g.tolist(),
                "halo_cells": halo_cells_g,
                "send": send_map_local,
                "recv": recv_map_local,
            }
        return halo_data

    @classmethod
    def create_local_meshes(
        cls,
        global_mesh: PolyMesh,
        n_parts: Optional[int] = None,
        cell_partitions: Optional[npt.NDArray[np.int_]] = None,
        partition_method: str = "metis",
        reorder_cells_strategy: Optional[str] = None,
        reorder_nodes_strategy: Optional[str] = None,
        store_order: bool = False,
    ) -> List["LocalMesh"]:
        """
        Partitions a global mesh and creates a list of local mesh objects.

        Args:
            global_mesh: The complete, unpartitioned PolyMesh object.
            n_parts: The desired number of partitions. Required if `cell_partitions`
                     is not provided.
            cell_partitions: An optional array specifying the partition ID for
                             each cell. If provided, `n_parts` is inferred.
            partition_method: The algorithm to use for partitioning if needed.
            reorder_cells_strategy: Optional strategy to reorder cells within each
                                    local mesh (e.g., 'rcm', 'bfs').
            reorder_nodes_strategy: Optional strategy to reorder nodes within each
                                    local mesh (e.g., 'rcm').
            store_order: If True, stores the original cell and node ordering for
                         later restoration. Defaults to False.

        Returns:
            A list of LocalMesh objects, one for each partition.
        """
        if global_mesh.cell_neighbors.size > 0:
            global_mesh.analyze_mesh()

        if cell_partitions is None:
            if n_parts is None or n_parts <= 0:
                raise ValueError(
                    "n_parts must be a positive integer or cell_partitions must be provided."
                )
            cell_partitions = partition_mesh(
                global_mesh, n_parts, method=partition_method
            )
            print_partition_summary(cell_partitions)
        else:
            n_parts = (
                int(np.max(cell_partitions) + 1) if cell_partitions.size > 0 else 0
            )

        if n_parts == 0:
            return []

        global_send_map = cls._find_send_candidates(global_mesh, cell_partitions)
        halo_indices = cls._compute_halo_indices(
            global_mesh, cell_partitions, global_send_map
        )

        local_meshes = []
        for rank in range(n_parts):
            if rank not in halo_indices or not halo_indices[rank]["owned_cells"]:
                # Skip ranks that have no owned cells
                continue

            final_store_order = (
                store_order
                or reorder_cells_strategy is not None
                or reorder_nodes_strategy is not None
            )
            local_mesh = LocalMesh.from_global_mesh(
                global_mesh, halo_indices[rank], rank, store_order=final_store_order
            )

            if reorder_cells_strategy:
                local_mesh.reorder_cells(strategy=reorder_cells_strategy, restore=False)
            if reorder_nodes_strategy:
                local_mesh.reorder_nodes(strategy=reorder_nodes_strategy, restore=False)

            local_mesh.analyze_mesh()
            local_meshes.append(local_mesh)

        return local_meshes
