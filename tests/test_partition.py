import os
import unittest

import numpy as np

from polymesh.core_mesh import CoreMesh
from polymesh.partition import partition_mesh, print_partition_summary


def make_test_mesh():
    """Creates a test mesh from a file."""
    msh_file = os.path.join(
        os.path.dirname(__file__), "..", "data", "sample_mixed_mesh.msh"
    )
    mesh = CoreMesh()
    mesh.read_gmsh(msh_file)
    return mesh


class TestPartitionMesh(unittest.TestCase):

    def setUp(self):
        self.tmp_path = "results/partition"
        os.makedirs(self.tmp_path, exist_ok=True)

    def test_metis_partitioning(self):
        """Test METIS partitioning."""
        mesh = make_test_mesh()
        n_parts = 4
        parts = partition_mesh(mesh, n_parts, method="metis")
        self.assertEqual(parts.shape[0], mesh.num_cells)
        self.assertEqual(len(np.unique(parts)), n_parts)
        mesh.plot(
            os.path.join(self.tmp_path, "mesh_partition_metis.png"),
            parts=parts,
        )
        print_partition_summary(parts)

    def test_hierarchical_partitioning(self):
        """Test hierarchical partitioning."""
        mesh = make_test_mesh()
        n_parts = 4
        parts = partition_mesh(mesh, n_parts, method="hierarchical")
        self.assertEqual(parts.shape[0], mesh.num_cells)
        self.assertEqual(len(np.unique(parts)), n_parts)
        mesh.plot(
            os.path.join(self.tmp_path, "mesh_partition_hierarchical.png"),
            parts=parts,
        )
        print_partition_summary(parts)


if __name__ == "__main__":
    unittest.main()
