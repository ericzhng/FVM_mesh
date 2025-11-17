import os
from typing import List
import unittest
import numpy as np
import copy

from fvm_mesh.polymesh import LocalMesh, PolyMesh, MeshPartitionManager
from fvm_mesh.polymesh.partition import partition_mesh


class TestLocalMesh(unittest.TestCase):
    def setUp(self):
        """
        Sets up a simple 3x3 quadrilateral mesh and partitions it for testing.
        """
        self.tmp_path = "results/localmesh"
        os.makedirs(self.tmp_path, exist_ok=True)

        nx, ny = 4, 4
        self.n_parts = 4
        self.global_mesh = PolyMesh.create_structured_quad_mesh(nx, ny)
        cell_partitions = partition_mesh(self.global_mesh, self.n_parts, method="metis")
        self.global_mesh.plot(f"{self.tmp_path}/global_mesh.png", cell_partitions)

        self.local_meshes = MeshPartitionManager.create_local_meshes(
            global_mesh=self.global_mesh,
            cell_partitions=cell_partitions,
            reorder_cells_strategy="",
            reorder_nodes_strategy="",
            store_order=True,  # Store ordering for restoration tests
        )

    def test_local_mesh_creation(self):
        """Tests the properties of the local mesh."""
        for rank, local_mesh in enumerate(self.local_meshes):
            local_mesh.plot(f"{self.tmp_path}/rank_{rank}_mesh.png")

            self.assertIsInstance(local_mesh, LocalMesh)
            self.assertEqual(local_mesh.rank, rank)
            self.assertEqual(local_mesh.num_owned_cells, 4)
            self.assertEqual(local_mesh.num_halo_cells, 4)
            self.assertEqual(local_mesh.n_cells, 8)  # owned + halo
            self.assertEqual(local_mesh.n_nodes, 15)  # owned + halo

    def test_symmetric_communication_maps(self):
        """Ensures that the send/recv maps are symmetric across partitions."""
        for i in range(self.n_parts):
            for j in range(i + 1, self.n_parts):
                mesh_i = self.local_meshes[i]
                mesh_j = self.local_meshes[j]

                # What rank i sends to rank j, rank j should expect to receive from rank i
                send_map_ij = mesh_i.send_map.get(j, [])
                recv_map_ji = mesh_j.recv_map.get(i, [])
                self.assertEqual(len(send_map_ij), len(recv_map_ji))

                # What rank j sends to rank i, rank i should expect to receive from rank j
                send_map_ji = mesh_j.send_map.get(i, [])
                recv_map_ij = mesh_i.recv_map.get(j, [])
                self.assertEqual(len(send_map_ji), len(recv_map_ij))

        # Verify send/recv maps
        n_parts = len(self.local_meshes)
        for i in range(n_parts):
            local_mesh = self.local_meshes[i]
            for neighbor_rank, send_indices in local_mesh.send_map.items():
                neighbor_mesh = next(
                    (m for m in self.local_meshes if m.rank == neighbor_rank), None
                )

                if neighbor_mesh is None:
                    self.fail(
                        f"Neighbor mesh with rank {neighbor_rank} not found among local_meshes"
                    )

                # Check for corresponding recv map
                self.assertIn(i, neighbor_mesh.recv_map)
                recv_indices = neighbor_mesh.recv_map[i]
                self.assertEqual(len(send_indices), len(recv_indices))

                # Check that the global cells are the same
                global_sent_cells = local_mesh.l2g_cells[send_indices]
                global_recv_cells = neighbor_mesh.l2g_cells[recv_indices]

                # Using assertCountEqual for order-independent comparison
                self.assertCountEqual(
                    global_sent_cells.tolist(), global_recv_cells.tolist()
                )

    def test_reorder_and_restore(self):
        """
        Tests that cell and node ordering can be restored after reordering.
        """
        local_mesh = self.local_meshes[0]
        self.assertIsNotNone(local_mesh)

        # --- Store original state for comparison ---
        original_l2g_cells = local_mesh.l2g_cells.copy()
        original_cell_node_conn = copy.deepcopy(local_mesh.cell_node_connectivity)
        original_send_map = copy.deepcopy(local_mesh.send_map)

        original_l2g_nodes = local_mesh.l2g_nodes.copy()
        original_node_coords = local_mesh.node_coords.copy()

        # --- Test Cell Reordering and Restoration ---
        local_mesh.reorder_cells(strategy="rcm", restore=False)
        self.assertTrue(local_mesh.use_reordered_cells)
        self.assertFalse(np.array_equal(original_l2g_cells, local_mesh.l2g_cells))
        if original_send_map:
            self.assertNotEqual(original_send_map, local_mesh.send_map)

        local_mesh.reorder_cells(restore=True)
        self.assertFalse(local_mesh.use_reordered_cells)
        np.testing.assert_array_equal(original_l2g_cells, local_mesh.l2g_cells)
        self.assertEqual(original_send_map, local_mesh.send_map)

        # --- Test Node Reordering and Restoration ---
        local_mesh.reorder_nodes(strategy="rcm", restore=False)
        self.assertTrue(local_mesh.use_reordered_nodes)
        self.assertFalse(np.array_equal(original_l2g_nodes, local_mesh.l2g_nodes))
        self.assertFalse(np.array_equal(original_node_coords, local_mesh.node_coords))

        local_mesh.reorder_nodes(restore=True)
        self.assertFalse(local_mesh.use_reordered_nodes)
        np.testing.assert_array_equal(original_l2g_nodes, local_mesh.l2g_nodes)
        np.testing.assert_array_equal(original_node_coords, local_mesh.node_coords)
        self.assertEqual(original_cell_node_conn, local_mesh.cell_node_connectivity)


if __name__ == "__main__":
    unittest.main()
