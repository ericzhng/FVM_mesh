import os
import unittest
import numpy as np

from polymesh import LocalMesh
from tests.common_meshes import create_3x3_quad_mesh_fixture


class TestLocalMesh(unittest.TestCase):
    def setUp(self):
        """
        Sets up a simple 3x3 quadrilateral mesh for testing.
        """
        self.tmp_path = "results/localmesh"
        os.makedirs(self.tmp_path, exist_ok=True)

        (
            self.global_mesh,
            self.parts,
            self.n_parts,
        ) = create_3x3_quad_mesh_fixture()

        self.global_mesh.plot(
            filepath=f"{self.tmp_path}/global_mesh.png", parts=self.parts
        )

    def test_local_mesh_creation_rank0(self):
        """Tests the creation and properties of the local mesh for rank 0."""
        rank = 0
        local_mesh = LocalMesh(self.global_mesh, self.parts, rank)

        # --- Verify cell properties ---
        self.assertEqual(local_mesh.rank, rank)
        self.assertEqual(local_mesh.num_owned_cells, 3)
        self.assertEqual(local_mesh.num_halo_cells, 3)
        self.assertEqual(local_mesh.num_cells, 6)

        # Owned cells (global IDs) should be [0, 1, 2]
        # Halo cells (global IDs) should be [3, 4, 5]
        np.testing.assert_array_equal(local_mesh.l2g_cells, [0, 1, 2, 3, 4, 5])
        self.assertEqual(local_mesh.g2l_cells, {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5})

        # --- Verify node properties ---
        self.assertEqual(local_mesh.num_nodes, 12)
        np.testing.assert_array_equal(local_mesh.l2g_nodes, np.arange(12))

    def test_local_mesh_creation_rank1(self):
        """Tests the creation and properties of the local mesh for rank 1."""
        rank = 1
        local_mesh = LocalMesh(self.global_mesh, self.parts, rank)

        # --- Verify cell properties ---
        self.assertEqual(local_mesh.rank, rank)
        self.assertEqual(local_mesh.num_owned_cells, 3)
        self.assertEqual(local_mesh.num_halo_cells, 6)
        self.assertEqual(local_mesh.num_cells, 9)

        # Owned cells (global IDs) should be [3, 4, 5]
        # Halo cells (global IDs) should be [0, 1, 2, 6, 7, 8]
        np.testing.assert_array_equal(local_mesh.l2g_cells, [3, 4, 5, 0, 1, 2, 6, 7, 8])
        self.assertEqual(
            local_mesh.g2l_cells, {3: 0, 4: 1, 5: 2, 0: 3, 1: 4, 2: 5, 6: 6, 7: 7, 8: 8}
        )

        # --- Verify node properties ---
        self.assertEqual(local_mesh.num_nodes, 16)
        np.testing.assert_array_equal(local_mesh.l2g_nodes, np.arange(16))

    def test_local_mesh_creation_rank2(self):
        """Tests the creation and properties of the local mesh for rank 2."""
        rank = 2
        local_mesh = LocalMesh(self.global_mesh, self.parts, rank)

        # --- Verify cell properties ---
        self.assertEqual(local_mesh.rank, rank)
        self.assertEqual(local_mesh.num_owned_cells, 3)
        self.assertEqual(local_mesh.num_halo_cells, 3)
        self.assertEqual(local_mesh.num_cells, 6)

        # Owned cells (global IDs) should be [6, 7, 8]
        # Halo cells (global IDs) should be [3, 4, 5]
        np.testing.assert_array_equal(local_mesh.l2g_cells, [6, 7, 8, 3, 4, 5])
        self.assertEqual(local_mesh.g2l_cells, {6: 0, 7: 1, 8: 2, 3: 3, 4: 4, 5: 5})

        # --- Verify node properties ---
        self.assertEqual(local_mesh.num_nodes, 12)
        expected_l2g = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        np.testing.assert_array_equal(local_mesh.l2g_nodes, expected_l2g)

    def test_symmetric_communication_maps(self):
        """Ensures that the send/recv maps are symmetric across partitions."""
        local_meshes = [
            LocalMesh(self.global_mesh, self.parts, rank=i) for i in range(self.n_parts)
        ]

        for i in range(self.n_parts):
            for j in range(i + 1, self.n_parts):
                # What rank i sends to rank j, rank j should expect to receive from rank i
                send_map_ij = local_meshes[i].send_map.get(j, [])
                recv_map_ji = local_meshes[j].recv_map.get(i, [])
                self.assertEqual(len(send_map_ij), len(recv_map_ji))

                # What rank j sends to rank i, rank i should expect to receive from rank j
                send_map_ji = local_meshes[j].send_map.get(i, [])
                recv_map_ij = local_meshes[i].recv_map.get(j, [])
                self.assertEqual(len(send_map_ji), len(recv_map_ij))


if __name__ == "__main__":
    unittest.main()
