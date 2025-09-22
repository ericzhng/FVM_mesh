import os
import unittest

import gmsh

from meshgen.geometry import Geometry


class TestGeometry(unittest.TestCase):
    def setUp(self):
        self.output_dir = "results/geometry"
        os.makedirs(self.output_dir, exist_ok=True)

    def tearDown(self):
        pass

    def test_create_polygon(self):
        geometry = Geometry(name="polygon")

        points = [
            (0, 0),
            (1, 0),
            (1, 1.4),
            (0, 1),
            (2, 1.5),
            (-0.5, 0.5),
            (1.5, 0.4),
            (2, 1),
        ]

        gmsh.initialize()
        gmsh.model.add("test_polygon")
        surface_tag = geometry.polygon(points, convex_hull=True)
        self.assertGreater(surface_tag, 0, "Surface tag should be positive.")

        # Verify that one surface was created
        surfaces = gmsh.model.getEntities(2)
        self.assertEqual(len(surfaces), 1, "There should be exactly one surface.")
        self.assertEqual(surfaces[0][1], surface_tag, "Surface tag should match.")

        geometry.plot(file_path=self.output_dir + "/test_polygon.png")
        self.assertTrue(
            os.path.exists(os.path.join(self.output_dir, "test_polygon.png"))
        )

        # For visual debugging, you can uncomment the following line.
        # gmsh.fltk.run()
        gmsh.finalize()

    def test_create_rectangle(self):
        geometry = Geometry(name="rectangle")

        gmsh.initialize()
        length, width = 5, 2
        mesh_size = 0.2

        gmsh.model.add("test_rectangle")
        surface_tag = geometry.rectangle(length, width, mesh_size=mesh_size)
        self.assertGreater(surface_tag, 0, "Surface tag should be positive.")

        # Verify that one surface was created
        surfaces = gmsh.model.getEntities(2)
        self.assertEqual(len(surfaces), 1, "There should be exactly one surface.")
        self.assertEqual(surfaces[0][1], surface_tag, "Surface tag should match.")

        geometry.plot(file_path=self.output_dir + "/test_rectangle.png")
        self.assertTrue(
            os.path.exists(os.path.join(self.output_dir, "test_rectangle.png"))
        )

        # For visual debugging, you can uncomment the following line.
        # gmsh.fltk.run()
        gmsh.finalize()

    def test_create_circle(self):
        geometry = Geometry(name="circle")

        gmsh.initialize()
        radius = 3
        mesh_size = 0.2

        gmsh.model.add("test_circle")
        surface_tag = geometry.circle(radius, mesh_size=mesh_size)
        self.assertGreater(surface_tag, 0, "Surface tag should be positive.")

        # Verify that one surface was created
        surfaces = gmsh.model.getEntities(2)
        self.assertEqual(len(surfaces), 1, "There should be exactly one surface.")
        self.assertEqual(surfaces[0][1], surface_tag, "Surface tag should match.")

        geometry.plot(file_path=self.output_dir + "/test_circle.png")
        self.assertTrue(
            os.path.exists(os.path.join(self.output_dir, "test_circle.png"))
        )

        # For visual debugging, you can uncomment the following line.
        # gmsh.fltk.run()
        gmsh.finalize()

    def test_create_triangle(self):
        geometry = Geometry(name="triangle")

        gmsh.initialize()
        gmsh.model.add("test_triangle")
        surface_tag = geometry.triangle((0, 0), (1, 0), (0.5, 1))
        self.assertGreater(surface_tag, 0, "Surface tag should be positive.")

        # Verify that one surface was created
        surfaces = gmsh.model.getEntities(2)
        self.assertEqual(len(surfaces), 1, "There should be exactly one surface.")
        self.assertEqual(surfaces[0][1], surface_tag, "Surface tag should match.")

        geometry.plot(file_path=self.output_dir + "/test_triangle.png")
        self.assertTrue(
            os.path.exists(os.path.join(self.output_dir, "test_triangle.png"))
        )

        # For visual debugging, you can uncomment the following line.
        # gmsh.fltk.run()
        gmsh.finalize()

    def test_create_ellipse(self):
        geometry = Geometry(name="ellipse")

        gmsh.initialize()
        gmsh.model.add("test_ellipse")
        surface_tag = geometry.ellipse(1, 0.5)
        self.assertGreater(surface_tag, 0, "Surface tag should be positive.")

        # Verify that one surface was created
        surfaces = gmsh.model.getEntities(2)
        self.assertEqual(len(surfaces), 1, "There should be exactly one surface.")
        self.assertEqual(surfaces[0][1], surface_tag, "Surface tag should match.")

        geometry.plot(file_path=self.output_dir + "/test_ellipse.png")
        self.assertTrue(
            os.path.exists(os.path.join(self.output_dir, "test_ellipse.png"))
        )

        # For visual debugging, you can uncomment the following line.
        # gmsh.fltk.run()
        gmsh.finalize()

    def test_create_rectangle_with_partitions(self):
        geometry = Geometry(name="rectangle_with_partitions")

        gmsh.initialize()
        length, width = 5, 2
        mesh_size = 0.2

        gmsh.model.add("test_rectangle_with_partitions")
        surface_tags = geometry.rectangle_with_partitions(
            length, width, mesh_size=mesh_size
        )
        self.assertEqual(len(surface_tags), 4, "Should return 4 surface tags.")

        # Verify that four surfaces were created
        surfaces = gmsh.model.occ.getEntities(2)
        self.assertEqual(len(surfaces), 4, "There should be exactly four surfaces.")

        geometry.plot(file_path=self.output_dir + "/test_rectangle_with_partitions.png")
        self.assertTrue(
            os.path.exists(
                os.path.join(self.output_dir, "test_rectangle_with_partitions.png")
            )
        )

        # For visual debugging, you can uncomment the following line.
        # gmsh.fltk.run()
        gmsh.finalize()


if __name__ == "__main__":
    unittest.main()
