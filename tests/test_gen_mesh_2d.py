import unittest
import os
import gmsh
from src.create_geometry_2d import Geometry2D
from src.gen_mesh_2d import Mesh2D


class TestMesh2D(unittest.TestCase):
    def setUp(self):
        """Set up for the test case."""
        self.output_dir = "test_output"
        os.makedirs(self.output_dir, exist_ok=True)
        gmsh.initialize()

        # Create a simple geometry for testing
        self.geom = Geometry2D(output_dir=self.output_dir)
        self.surface_tag = self.geom.rectangle(length=1, width=1, mesh_size=0.5)
        gmsh.model.occ.synchronize()

    def tearDown(self):
        """Tear down the test case."""
        gmsh.finalize()
        # Clean up generated files
        # for f in os.listdir(self.output_dir):
        #     os.remove(os.path.join(self.output_dir, f))
        # os.rmdir(self.output_dir)

    def test_triangular_mesh_generation(self):
        """Test the generation of a triangular mesh."""
        mesher = Mesh2D(surface_tags=self.surface_tag, output_dir=self.output_dir)
        mesh_filename = "triangular_mesh.msh"
        plot_filename = "triangular_mesh.png"
        mesher.generate(mesh_type="triangular", char_length=0.1, filename=mesh_filename)

        # Check if the mesh and plot files are created
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, mesh_filename)))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, plot_filename)))

        # For visual debugging, you can uncomment the following line.
        gmsh.fltk.run()

    def test_structured_mesh_generation(self):
        """Test the generation of a structured mesh."""
        mesher = Mesh2D(surface_tags=self.surface_tag, output_dir=self.output_dir)
        mesh_filename = "structured_mesh.msh"
        plot_filename = "structured_mesh.png"
        mesher.generate(mesh_type="structured", char_length=0.2, filename=mesh_filename)

        # Check if the mesh and plot files are created
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, mesh_filename)))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, plot_filename)))

        # For visual debugging, you can uncomment the following line.
        gmsh.fltk.run()

    def test_invalid_mesh_type(self):
        """Test that an invalid mesh type raises a ValueError."""
        mesher = Mesh2D(surface_tags=self.surface_tag, output_dir=self.output_dir)
        with self.assertRaises(ValueError):
            mesher.generate(mesh_type="invalid_type")


if __name__ == "__main__":
    unittest.main()
