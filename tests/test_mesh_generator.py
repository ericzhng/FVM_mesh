import os
import unittest

import gmsh

from fvm_mesh.meshgen.geometry import Geometry
from fvm_mesh.meshgen.mesh_generator import MeshGenerator


class TestMesh2D(unittest.TestCase):
    def setUp(self):
        """Set up for the test case."""
        self.output_dir = "results/generation"
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
        mesher.generate(
            mesh_params=mesh_params,
            filename=mesh_filename,
            show_nodes=True,
            show_cells=True,
        )

        # Check if the mesh and plot files are created
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, mesh_filename)))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, plot_filename)))

    def test_structured_mesh_generation(self):
        """Test the generation of a structured mesh."""
        projName = "rect_structured_mesh"
        gmsh.model.add(projName)

        geom = Geometry(projName)
        surface_tag = geom.rectangle(length=100, width=100, mesh_size=2)

        mesher = MeshGenerator(surface_tags=surface_tag, output_dir=self.output_dir)
        mesh_filename = "structured_mesh.msh"
        plot_filename = "structured_mesh.png"
        mesh_params = {surface_tag: {"mesh_type": "structured", "char_length": 5}}
        mesher.generate(
            mesh_params=mesh_params,
            filename=mesh_filename,
            show_nodes=True,
            show_cells=True,
        )

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
        mesher.generate(
            mesh_params=mesh_params,
            filename=mesh_filename,
            show_nodes=True,
            show_cells=True,
        )

        # Check if the mesh and plot files are created
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, mesh_filename)))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, plot_filename)))

    def test_mixed_mesh_generation(self):
        """Test the generation of a mixed structured and triangular mesh."""
        # 1. Create geometry with partitions, setting the global mesh size.
        projName = "mixed_mesh"
        gmsh.model.add(projName)

        geom = Geometry(projName)
        surfaces = geom.rectangle_with_partitions(length=10, width=10, mesh_size=0.5)

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
        mesher.generate(
            mesh_params=mesh_params,
            filename=mesh_filename,
            show_nodes=True,
            show_cells=True,
        )

        # 5. Check if all files are created
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, mesh_filename)))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, plot_filename)))

    def test_triangular_mesh_on_circle(self):
        """Test the generation of a unstructured mesh on a circle."""
        projName = "unstructured_mesh_on_circle"
        gmsh.model.add(projName)

        geom = Geometry(projName)
        surface_tag = geom.circle(radius=1, mesh_size=0.5)

        mesher = MeshGenerator(surface_tags=surface_tag, output_dir=self.output_dir)
        mesh_filename = "circle_mesh.msh"
        plot_filename = "circle_mesh.png"
        mesh_params = {surface_tag: {"mesh_type": "tri", "char_length": 0.2}}
        mesher.generate(
            mesh_params=mesh_params,
            filename=mesh_filename,
            show_nodes=True,
            show_cells=True,
        )

        # Check if the mesh and plot files are created
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, mesh_filename)))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, plot_filename)))

    def test_invalid_mesh_type(self):
        """Test that an invalid mesh type raises a ValueError."""
        geom = Geometry("invalid_mesh_type")
        surface_tag = geom.rectangle(length=1, width=1, mesh_size=0.5)

        mesher = MeshGenerator(surface_tags=surface_tag, output_dir=self.output_dir)
        mesh_params = {surface_tag: {"mesh_type": "invalid_type"}}
        with self.assertRaises(ValueError):
            mesher.generate(mesh_params=mesh_params)

    def test_physical_group_setup(self):
        """Test the automatic setup of physical groups."""
        projName = "physical_group_test"
        gmsh.model.add(projName)

        geom = Geometry(projName)
        surface_tag = geom.rectangle(length=1, width=1, mesh_size=0.1)

        # Get boundary curves
        boundary_curves = [c[1] for c in gmsh.model.getBoundary([(2, surface_tag)])]
        self.assertEqual(len(boundary_curves), 4)

        # Manually assign one curve to a physical group
        manual_group_name = "inlet"
        gmsh.model.addPhysicalGroup(1, [boundary_curves[0]], name=manual_group_name)

        # Use the mesh generator
        mesher = MeshGenerator(surface_tags=surface_tag, output_dir=self.output_dir)
        mesh_params = {surface_tag: {"mesh_type": "tri"}}
        mesher.generate(mesh_params=mesh_params, filename="physical_group_test.msh")

        # --- Verification ---

        # 1. Verify 2D "fluid" group
        phys_groups_2d = gmsh.model.getPhysicalGroups(2)
        self.assertEqual(len(phys_groups_2d), 1, "Should be one 2D physical group.")
        fluid_group_tag = phys_groups_2d[0][1]
        fluid_group_name = gmsh.model.getPhysicalName(2, fluid_group_tag)
        self.assertEqual(fluid_group_name, "fluid")
        fluid_entities = gmsh.model.getEntitiesForPhysicalGroup(2, fluid_group_tag)
        self.assertIn(surface_tag, fluid_entities)

        # 2. Verify 1D groups ("inlet" and "unnamed")
        phys_groups_1d = gmsh.model.getPhysicalGroups(1)
        self.assertEqual(len(phys_groups_1d), 2, "Should be two 1D physical groups.")

        group_names = [gmsh.model.getPhysicalName(1, tag) for dim, tag in phys_groups_1d]
        self.assertIn("inlet", group_names)
        self.assertIn("unnamed", group_names)

        # 3. Verify content of "unnamed" group
        unnamed_tag = -1
        for dim, tag in phys_groups_1d:
            if gmsh.model.getPhysicalName(dim, tag) == "unnamed":
                unnamed_tag = tag
                break
        
        unnamed_entities = gmsh.model.getEntitiesForPhysicalGroup(1, unnamed_tag)
        self.assertEqual(len(unnamed_entities), 3)
        
        # Check that the unassigned curves are in the "unnamed" group
        unassigned_curves = set(boundary_curves[1:])
        self.assertEqual(set(unnamed_entities), unassigned_curves)

        # 4. Verify content of "inlet" group
        inlet_tag = -1
        for dim, tag in phys_groups_1d:
            if gmsh.model.getPhysicalName(dim, tag) == "inlet":
                inlet_tag = tag
                break
        inlet_entities = gmsh.model.getEntitiesForPhysicalGroup(1, inlet_tag)
        self.assertEqual(len(inlet_entities), 1)
        self.assertEqual(inlet_entities[0], boundary_curves[0])


if __name__ == "__main__":
    unittest.main()
