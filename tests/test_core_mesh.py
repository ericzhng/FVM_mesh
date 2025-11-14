import unittest
import os
import numpy as np

from fvm_mesh.polymesh.core_mesh import CoreMesh


class TestCoreMesh(unittest.TestCase):

    def setUp(self):
        """Set up a CoreMesh instance by reading a sample mesh file."""
        self.output_dir = "results/coremesh"
        os.makedirs(self.output_dir, exist_ok=True)

        self.mesh = CoreMesh()
        self.test_msh_file = os.path.join(
            os.path.dirname(__file__), "..", "data", "sample_rect_mesh.msh"
        )
        self.mesh.read_gmsh(self.test_msh_file)
        self.mesh.analyze_mesh()
        self.mesh.plot(os.path.join(self.output_dir, "core_mesh_plot.png"))

    def test_read_gmsh(self):
        """Test if the Gmsh file is read correctly."""
        self.assertEqual(self.mesh.dimension, 2)
        self.assertEqual(self.mesh.num_nodes, 504)
        self.assertEqual(self.mesh.node_coords.shape, (504, 3))
        self.assertEqual(self.mesh.num_cells, 463)
        self.assertEqual(len(self.mesh.cell_connectivity), 463)

    def test_extract_neighbors(self):
        """Test the neighbor extraction functionality."""
        self.mesh._extract_cell_faces()
        self.mesh._extract_cell_neighbors()
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

    def test_boundary_faces(self):
        """Test boundary face and tag reading."""
        self.assertGreater(len(self.mesh.boundary_faces_nodes), 0)
        self.assertGreater(len(self.mesh.boundary_faces_tags), 0)
        self.assertEqual(
            len(self.mesh.boundary_faces_nodes), len(self.mesh.boundary_faces_tags)
        )
        # Check that tags are stored as a list of lists
        self.assertIsInstance(self.mesh.boundary_faces_tags, np.ndarray)


if __name__ == "__main__":
    unittest.main()
