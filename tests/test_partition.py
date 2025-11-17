import os
import unittest

import numpy as np

from fvm_mesh.polymesh.partition import partition_mesh, print_partition_summary
from fvm_mesh.polymesh.poly_mesh import PolyMesh


class TestPartitionMesh(unittest.TestCase):

    def setUp(self):
        self.tmp_path = "results/partition"
        os.makedirs(self.tmp_path, exist_ok=True)

    def test_metis_partitioning(self):
        """Test METIS partitioning."""
        mesh = PolyMesh.create_structured_quad_mesh(15, 15)
        n_parts = 3
        parts = partition_mesh(mesh, n_parts, method="metis")
        self.assertEqual(parts.shape[0], mesh.n_cells)
        self.assertEqual(len(np.unique(parts)), n_parts)
        mesh.plot(
            os.path.join(self.tmp_path, "mesh_partition_metis.png"),
            parts=parts,
        )
        print_partition_summary(parts)

    def test_hierarchical_partitioning(self):
        """Test hierarchical partitioning."""
        mesh = PolyMesh.create_structured_quad_mesh(15, 15)
        n_parts = 3
        parts = partition_mesh(mesh, n_parts, method="hierarchical")
        self.assertEqual(parts.shape[0], mesh.n_cells)
        self.assertEqual(len(np.unique(parts)), n_parts)
        mesh.plot(
            os.path.join(self.tmp_path, "mesh_partition_hierarchical.png"),
            parts=parts,
        )
        print_partition_summary(parts)


if __name__ == "__main__":
    unittest.main()
