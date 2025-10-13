import unittest
import os
import numpy as np

from polymesh.core_mesh import CoreMesh


class TestCoreMesh(unittest.TestCase):

    def setUp(self):
        """Set up a CoreMesh instance by reading a sample mesh file."""
        self.mesh = CoreMesh()
        self.test_msh_file = os.path.join(
            os.path.dirname(__file__), "..", "data", "sample_mixed_mesh.msh"
        )
        self.mesh.read_gmsh(self.test_msh_file)

    def test_read_gmsh(self):
        """Test if the Gmsh file is read correctly."""
        self.assertEqual(self.mesh.dimension, 2)
        self.assertEqual(self.mesh.num_nodes, 1210)
        self.assertEqual(self.mesh.node_coords.shape, (1210, 3))
        self.assertEqual(self.mesh.num_cells, 1384)
        self.assertEqual(len(self.mesh.cell_connectivity), 1384)

    def test_extract_neighbors(self):
        """Test the neighbor extraction functionality."""
        self.mesh._extract_neighbors()
        self.assertIsNotNone(self.mesh.cell_neighbors)
        self.assertEqual(self.mesh.cell_neighbors.shape[0], self.mesh.num_cells)
        # Check if there are any valid neighbors (not -1)
        self.assertTrue(np.any(self.mesh.cell_neighbors != -1))

    def test_compute_centroids(self):
        """Test the cell centroid computation."""
        self.mesh._compute_centroids()
        self.assertIsNotNone(self.mesh.cell_centroids)
        self.assertEqual(self.mesh.cell_centroids.shape, (self.mesh.num_cells, 3))
        # Check if centroids are within the mesh bounds
        min_coords = np.min(self.mesh.node_coords, axis=0)
        max_coords = np.max(self.mesh.node_coords, axis=0)
        self.assertTrue(np.all(self.mesh.cell_centroids >= min_coords))
        self.assertTrue(np.all(self.mesh.cell_centroids <= max_coords))


if __name__ == "__main__":
    unittest.main()
