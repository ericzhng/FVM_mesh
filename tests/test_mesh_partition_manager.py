import os
import unittest
import numpy as np

from fvm_mesh.polymesh import PolyMesh, MeshPartitionManager
from fvm_mesh.polymesh.partition import partition_mesh


class TestMeshPartitionManager(unittest.TestCase):
    def setUp(self):
        """
        Sets up a simple 3x3 quadrilateral mesh for testing.
        """
        self.tmp_path = "results/partition_manager"
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

    def test_create_local_meshes_from_fixture(self):
        """
        Tests the creation of local meshes from a predefined fixture.
        """
        self.assertEqual(len(self.local_meshes), self.n_parts)

        total_owned_cells = sum(mesh.num_owned_cells for mesh in self.local_meshes)
        self.assertEqual(total_owned_cells, self.global_mesh.n_cells)

    def test_create_local_meshes_from_file(self):
        """
        Tests the creation of local meshes from a GMSH file, including reordering.
        """
        n_parts = 4
        partition_method = "metis"

        test_msh_file = os.path.join(
            os.path.dirname(__file__), "..", "data", "sample_rect_mesh.msh"
        )
        global_mesh = PolyMesh().from_gmsh(test_msh_file)
        cell_partitions = partition_mesh(global_mesh, n_parts, method=partition_method)
        global_mesh.plot(f"{self.tmp_path}/global_mesh.png", cell_partitions)

        local_meshes = MeshPartitionManager.create_local_meshes(
            global_mesh,
            cell_partitions=cell_partitions,
            reorder_cells_strategy="rcm",
            reorder_nodes_strategy="rcm",
        )
        self.assertEqual(len(local_meshes), n_parts)

        total_owned_cells = sum(mesh.num_owned_cells for mesh in local_meshes)
        self.assertEqual(total_owned_cells, global_mesh.n_cells)

        for local_mesh in local_meshes:
            self.assertTrue(local_mesh.use_reordered_cells)
            self.assertTrue(local_mesh.use_reordered_nodes)
            local_mesh.plot(
                f"{self.tmp_path}sample_rect_mesh_local_{local_mesh.rank}.png"
            )

    def test_partition_with_n_parts_one(self):
        """
        Tests partitioning with n_parts=1, which should result in a single
        local mesh with no halo.
        """
        local_meshes = MeshPartitionManager.create_local_meshes(
            global_mesh=self.global_mesh, n_parts=1
        )
        self.assertEqual(len(local_meshes), 1)
        self.assertEqual(local_meshes[0].num_owned_cells, self.global_mesh.n_cells)
        self.assertEqual(local_meshes[0].num_halo_cells, 0)
        self.assertEqual(len(local_meshes[0].send_map), 0)
        self.assertEqual(len(local_meshes[0].recv_map), 0)

    def test_invalid_arguments_raise_error(self):
        """
        Tests that providing invalid arguments to create_local_meshes raises a ValueError.
        """
        with self.assertRaises(ValueError):
            MeshPartitionManager.create_local_meshes(
                global_mesh=self.global_mesh, n_parts=0
            )

        with self.assertRaises(ValueError):
            MeshPartitionManager.create_local_meshes(
                global_mesh=self.global_mesh, n_parts=None, cell_partitions=None
            )


if __name__ == "__main__":
    unittest.main()
