import unittest
import numpy as np

from polymesh import PolyMesh, LocalMesh


class TestLocalMesh(unittest.TestCase):

    def setUp(self):
        """
        Sets up a simple 2x2 quadrilateral mesh for testing.

        Mesh Layout:
        Nodes (9 total):
        6--7--8
        |  |  |
        3--4--5
        |  |  |
        0--1--2

        Cells (4 total):
        Cell 2: [3, 4, 7, 6]
        Cell 3: [4, 5, 8, 7]
        Cell 0: [0, 1, 4, 3]
        Cell 1: [1, 2, 5, 4]

        Partitioning (2 parts):
        - Rank 0 owns cells [0, 2]
        - Rank 1 owns cells [1, 3]
        """
        self.global_mesh = PolyMesh()

        # Define nodes
        self.global_mesh.node_coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],  # Nodes 0, 1, 2
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [2.0, 1.0, 0.0],  # Nodes 3, 4, 5
                [0.0, 2.0, 0.0],
                [1.0, 2.0, 0.0],
                [2.0, 2.0, 0.0],  # Nodes 6, 7, 8
            ]
        )

        # Define cells
        self.global_mesh.cell_connectivity = [
            [0, 1, 4, 3],  # Cell 0
            [1, 2, 5, 4],  # Cell 1
            [3, 4, 7, 6],  # Cell 2
            [4, 5, 8, 7],  # Cell 3
        ]

        self.global_mesh.dimension = 2
        self.global_mesh.num_nodes = 9
        self.global_mesh.num_cells = 4

        self.global_mesh.plot(filepath="global_mesh.png")

        # # Analyze the mesh to generate neighbors, etc.
        # self.global_mesh.extract_neighbors()

        # Define the partitioning scheme
        self.parts = np.array([0, 1, 0, 1])
        self.n_parts = 2

    def test_local_mesh_creation_rank0(self):
        """Tests the creation and properties of the local mesh for rank 0."""
        rank = 0
        local_mesh = LocalMesh(self.global_mesh, self.parts, rank)

        # --- Verify cell properties ---
        self.assertEqual(local_mesh.rank, rank)
        self.assertEqual(local_mesh.num_owned_cells, 2)
        self.assertEqual(local_mesh.num_halo_cells, 2)
        self.assertEqual(local_mesh.num_cells, 4)

        # Owned cells (global IDs) should be [0, 2]
        # Halo cells (global IDs) should be [1, 3]
        np.testing.assert_array_equal(local_mesh.l2g_cells, [0, 2, 1, 3])
        self.assertEqual(local_mesh.g2l_cells, {0: 0, 2: 1, 1: 2, 3: 3})

        # --- Verify node properties ---
        # All 9 nodes are needed for this partition
        self.assertEqual(local_mesh.num_nodes, 9)
        np.testing.assert_array_equal(local_mesh.l2g_nodes, np.arange(9))

        # --- Verify communication maps ---
        # Rank 0 should send its two owned cells (local indices 0, 1) to rank 1
        self.assertIn(1, local_mesh.send_map)
        self.assertEqual(sorted(local_mesh.send_map[1]), [0, 1])

        # Rank 0 should expect to receive two halo cells from rank 1
        # These will be stored at local halo indices 0 and 1
        self.assertIn(1, local_mesh.recv_map)
        self.assertEqual(sorted(local_mesh.recv_map[1]), [0, 1])

    def test_local_mesh_creation_rank1(self):
        """Tests the creation and properties of the local mesh for rank 1."""
        rank = 1
        local_mesh = LocalMesh(self.global_mesh, self.parts, rank)

        # --- Verify cell properties ---
        self.assertEqual(local_mesh.rank, rank)
        self.assertEqual(local_mesh.num_owned_cells, 2)
        self.assertEqual(local_mesh.num_halo_cells, 2)
        self.assertEqual(local_mesh.num_cells, 4)

        # Owned cells (global IDs) should be [1, 3]
        # Halo cells (global IDs) should be [0, 2]
        np.testing.assert_array_equal(local_mesh.l2g_cells, [1, 3, 0, 2])
        self.assertEqual(local_mesh.g2l_cells, {1: 0, 3: 1, 0: 2, 2: 3})

        # --- Verify node properties ---
        # All 9 nodes are needed for this partition
        self.assertEqual(local_mesh.num_nodes, 9)
        np.testing.assert_array_equal(local_mesh.l2g_nodes, np.arange(9))

        # --- Verify communication maps ---
        # Rank 1 should send its two owned cells (local indices 0, 1) to rank 0
        self.assertIn(0, local_mesh.send_map)
        self.assertEqual(sorted(local_mesh.send_map[0]), [0, 1])

        # Rank 1 should expect to receive two halo cells from rank 0
        # These will be stored at local halo indices 0 and 1
        self.assertIn(0, local_mesh.recv_map)
        self.assertEqual(sorted(local_mesh.recv_map[0]), [0, 1])

    def test_symmetric_communication_maps(self):
        """Ensures that the send/recv maps are symmetric across partitions."""
        local_mesh_0 = LocalMesh(self.global_mesh, self.parts, rank=0)
        local_mesh_1 = LocalMesh(self.global_mesh, self.parts, rank=1)

        # What rank 0 sends to rank 1, rank 1 should expect to receive from rank 0
        self.assertEqual(
            len(local_mesh_0.send_map.get(1, [])), len(local_mesh_1.recv_map.get(0, []))
        )

        # What rank 1 sends to rank 0, rank 0 should expect to receive from rank 1
        self.assertEqual(
            len(local_mesh_1.send_map.get(0, [])), len(local_mesh_0.recv_map.get(1, []))
        )


if __name__ == "__main__":
    unittest.main()
