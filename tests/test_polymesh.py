import unittest
import os
import numpy as np

from fvm_mesh.polymesh.poly_mesh import PolyMesh


class TestPolyMesh(unittest.TestCase):
    """Unit tests for the PolyMesh class based on a structured mesh."""

    @classmethod
    def setUpClass(cls):
        """Set up a structured quad mesh for all tests."""
        cls.output_dir = "results/polymesh"
        os.makedirs(cls.output_dir, exist_ok=True)

        cls.nx, cls.ny = 5, 5
        cls.mesh = PolyMesh.create_structured_quad_mesh(cls.nx, cls.ny)

    def test_creation_and_properties(self):
        """Test creating a mesh and its basic properties."""
        self.assertIsInstance(self.mesh, PolyMesh)
        self.assertEqual(self.mesh.dimension, 2)
        self.assertEqual(self.mesh.n_nodes, (self.nx + 1) * (self.ny + 1))
        self.assertEqual(self.mesh.n_cells, self.nx * self.ny)
        self.assertEqual(
            self.mesh.node_coords.shape, ((self.nx + 1) * (self.ny + 1), 3)
        )
        self.assertEqual(len(self.mesh.cell_node_connectivity), self.nx * self.ny)
        self.assertEqual(self.mesh.cell_element_types.shape, (self.nx * self.ny,))
        self.assertEqual(self.mesh.element_type_properties[0]["name"], "Quad 4")

    def test_face_topology(self):
        """Test the computation of face-based topology (neighbors, etc.)."""
        self.assertIsNotNone(self.mesh.cell_neighbors)
        self.assertEqual(self.mesh.cell_neighbors.shape, (self.nx * self.ny, 4))

        # Check for both interior and boundary faces
        self.assertTrue(np.any(self.mesh.cell_neighbors != -1))
        self.assertTrue(np.any(self.mesh.cell_neighbors == -1))

        # Check that face tags are correctly assigned for boundaries
        boundary_face_mask = self.mesh.cell_neighbors == -1
        self.assertTrue(np.any(self.mesh.cell_face_tags[boundary_face_mask] > 0))

        # Interior faces should have a tag of 0
        interior_face_mask = self.mesh.cell_neighbors != -1
        self.assertTrue(np.all(self.mesh.cell_face_tags[interior_face_mask] == 0))

    def test_geometric_properties(self):
        """Test the computation of various geometric properties."""
        # Centroids
        self.assertEqual(self.mesh.cell_centroids.shape, (self.nx * self.ny, 3))
        min_coords = np.min(self.mesh.node_coords, axis=0)
        max_coords = np.max(self.mesh.node_coords, axis=0)
        self.assertTrue(np.all(self.mesh.cell_centroids >= min_coords))
        self.assertTrue(np.all(self.mesh.cell_centroids <= max_coords))

        # Volumes (Areas in 2D)
        self.assertEqual(self.mesh.cell_volumes.shape, (self.nx * self.ny,))
        self.assertTrue(np.all(self.mesh.cell_volumes > 0))
        # For a unit quad mesh, all cell areas should be 1.0
        self.assertTrue(np.allclose(self.mesh.cell_volumes, 1.0))

        # Face Areas
        self.assertEqual(self.mesh.cell_face_areas.shape, (self.nx * self.ny, 4))
        # All face areas (lengths) should be 1.0
        valid_areas = self.mesh.cell_face_areas[self.mesh.cell_face_areas > -1]
        self.assertTrue(np.allclose(valid_areas, 1.0))

        # Face Normals
        self.assertEqual(self.mesh.cell_face_normals.shape, (self.nx * self.ny, 4, 3))
        norms = np.linalg.norm(self.mesh.cell_face_normals, axis=2)
        self.assertTrue(np.allclose(norms[norms > 0.1], 1.0))

    def test_boundary_data(self):
        """Test the processing of boundary face data."""
        self.assertEqual(len(self.mesh.boundary_patch_map), 4)
        self.assertIn("bottom", self.mesh.boundary_patch_map)
        self.assertIn("right", self.mesh.boundary_patch_map)
        self.assertIn("top", self.mesh.boundary_patch_map)
        self.assertIn("left", self.mesh.boundary_patch_map)

        num_boundary_faces = 2 * self.nx + 2 * self.ny
        self.assertEqual(self.mesh.boundary_face_nodes.shape[0], num_boundary_faces)
        self.assertEqual(self.mesh.boundary_face_tags.shape[0], num_boundary_faces)

    def test_analysis_and_output(self):
        """Test the full analysis flag and output generation."""
        self.assertTrue(self.mesh._is_analyzed)

        import io
        from contextlib import redirect_stdout

        with redirect_stdout(io.StringIO()) as f:
            self.mesh.print_summary()

        summary_output = f.getvalue()
        self.assertIn("Mesh Analysis Report", summary_output)
        self.assertIn("Number of Cells:", summary_output)

        plot_path = os.path.join(self.output_dir, "poly_mesh_plot.png")
        self.mesh.plot(filepath=plot_path)
        self.assertTrue(os.path.exists(plot_path))

    def test_error_handling(self):
        """Test error handling for invalid operations."""
        empty_mesh = PolyMesh()
        with self.assertRaises(RuntimeError):
            empty_mesh.analyze_mesh()


if __name__ == "__main__":
    unittest.main()
