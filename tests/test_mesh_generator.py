import unittest
import os
import gmsh
from src.geometry import Geometry
from src.mesh_generator import MeshGenerator


class TestMesh2D(unittest.TestCase):
    def setUp(self):
        """Set up for the test case."""
        self.output_dir = "trunk"
        os.makedirs(self.output_dir, exist_ok=True)
        gmsh.initialize()

    def tearDown(self):
        """Tear down the test case."""
        gmsh.finalize()

    def test_triangular_mesh_generation(self):
        """Test the generation of a triangular mesh."""
        projName = "rect_grid_mesh"
        gmsh.model.add(projName)

        geom = Geometry(projName)
        surface_tag = geom.rectangle(length=1, width=1, mesh_size=0.1)

        mesher = MeshGenerator(surface_tags=surface_tag, output_dir=self.output_dir)
        mesh_filename = "triangular_mesh.msh"
        plot_filename = "triangular_mesh.png"
        mesh_params = {surface_tag: {"mesh_type": "tri", "char_length": 0.1}}
        mesher.generate(mesh_params=mesh_params, filename=mesh_filename)

        # Check if the mesh and plot files are created
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, mesh_filename)))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, plot_filename)))

    def test_structured_mesh_generation(self):
        """Test the generation of a structured mesh."""
        projName = "rect_structured_mesh"
        gmsh.model.add(projName)

        geom = Geometry(projName)
        surface_tag = geom.rectangle(length=1, width=1, mesh_size=0.2)

        mesher = MeshGenerator(surface_tags=surface_tag, output_dir=self.output_dir)
        mesh_filename = "structured_mesh.msh"
        plot_filename = "structured_mesh.png"
        mesh_params = {surface_tag: {"mesh_type": "structured", "char_length": 0.2}}
        mesher.generate(mesh_params=mesh_params, filename=mesh_filename)

        # Check if the mesh and plot files are created
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, mesh_filename)))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, plot_filename)))

    def test_quad_mesh_generation(self):
        """Test the generation of a quad mesh."""
        projName = "rect_quad_mesh"
        gmsh.model.add(projName)

        geom = Geometry()
        surface_tag = geom.rectangle(length=1, width=1, mesh_size=0.1)

        mesher = MeshGenerator(surface_tags=surface_tag, output_dir=self.output_dir)
        mesh_filename = "quad_mesh.msh"
        plot_filename = "quad_mesh.png"
        mesh_params = {surface_tag: {"mesh_type": "quads"}}
        mesher.generate(mesh_params=mesh_params, filename=mesh_filename)

        # Check if the mesh and plot files are created
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, mesh_filename)))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, plot_filename)))

    def test_invalid_mesh_type(self):
        """Test that an invalid mesh type raises a ValueError."""
        geom = Geometry()
        surface_tag = geom.rectangle(length=1, width=1, mesh_size=0.5)

        mesher = MeshGenerator(surface_tags=surface_tag, output_dir=self.output_dir)
        mesh_params = {surface_tag: {"mesh_type": "invalid_type"}}
        with self.assertRaises(ValueError):
            mesher.generate(mesh_params=mesh_params)

    def test_mixed_mesh_generation(self):
        """Test the generation of a mixed structured and triangular mesh."""
        # 1. Create geometry with partitions, setting the global mesh size.
        projName = "mixed_mesh"
        gmsh.model.add(projName)

        geom = Geometry(projName)
        surfaces = geom.rectangle_with_partitions(length=2, width=1, mesh_size=0.2)

        self.assertEqual(len(surfaces), 4)

        # 2. We will mesh all four surfaces for this test
        mesher = MeshGenerator(surface_tags=surfaces, output_dir=self.output_dir)

        # 3. Define mesh parameters for all four surfaces
        mesh_params = {
            surfaces[0]: {"mesh_type": "structured", "char_length": 0.2},
            surfaces[1]: {"mesh_type": "quads"},
            surfaces[2]: {"mesh_type": "tri"},
            surfaces[3]: {"mesh_type": "quads"},
        }

        mesh_filename = "mixed_mesh_all_surfaces.msh"
        plot_filename = "mixed_mesh_all_surfaces.png"

        # 4. Generate the mesh
        mesher.generate(mesh_params=mesh_params, filename=mesh_filename)

        # 5. Check if all files are created
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, mesh_filename)))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, plot_filename)))

    def test_plot_with_labels(self):
        """Test the plot generation with node and cell labels."""
        projName = "plot_with_labels"
        gmsh.model.add(projName)

        geom = Geometry(projName)
        surface_tag = geom.rectangle(length=1, width=1, mesh_size=0.3)

        mesher = MeshGenerator(surface_tags=surface_tag, output_dir=self.output_dir)
        mesh_filename = "plot_with_labels.msh"
        plot_filename = "plot_with_labels.png"
        mesh_params = {surface_tag: {"mesh_type": "tri", "char_length": 0.3}}
        mesher.generate(
            mesh_params=mesh_params,
            filename=mesh_filename,
            show_nodes=True,
            show_cells=True,
        )

        # Check if the mesh and plot files are created
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, mesh_filename)))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, plot_filename)))


if __name__ == "__main__":
    unittest.main()