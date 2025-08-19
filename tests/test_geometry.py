import gmsh
import unittest
import os
from src.geometry import Geometry


class TestGeometry(unittest.TestCase):
    def setUp(self):
        self.output_dir = "./trunk"
        self.geometry = Geometry(output_dir=self.output_dir)

    def tearDown(self):
        pass

    def test_create_polygon(self):
        gmsh.initialize()
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

        gmsh.model.add("test_polygon")
        surface_tag = self.geometry.polygon(points, convex_hull=True)
        self.assertGreater(surface_tag, 0, "Surface tag should be positive.")

        # Verify that one surface was created
        surfaces = gmsh.model.getEntities(2)
        self.assertEqual(len(surfaces), 1, "There should be exactly one surface.")
        self.assertEqual(surfaces[0][1], surface_tag, "Surface tag should match.")

        self.geometry.plot(file_name="test_polygon.png")
        self.assertTrue(
            os.path.exists(os.path.join(self.output_dir, "test_polygon.png"))
        )

        # For visual debugging, you can uncomment the following line.
        gmsh.fltk.run()
        gmsh.finalize()

    def test_create_rectangle(self):
        gmsh.initialize()
        length, width = 5, 2
        mesh_size = 0.2

        gmsh.model.add("test_rectangle")
        surface_tag = self.geometry.rectangle(length, width, mesh_size=mesh_size)
        self.assertGreater(surface_tag, 0, "Surface tag should be positive.")

        # Verify that one surface was created
        surfaces = gmsh.model.getEntities(2)
        self.assertEqual(len(surfaces), 1, "There should be exactly one surface.")
        self.assertEqual(surfaces[0][1], surface_tag, "Surface tag should match.")

        self.geometry.plot(file_name="test_rectangle.png")
        self.assertTrue(
            os.path.exists(os.path.join(self.output_dir, "test_rectangle.png"))
        )

        # For visual debugging, you can uncomment the following line.
        gmsh.fltk.run()
        gmsh.finalize()

    def test_create_circle(self):
        gmsh.initialize()
        radius = 3
        mesh_size = 0.2

        gmsh.model.add("test_circle")
        surface_tag = self.geometry.circle(radius, mesh_size=mesh_size)
        self.assertGreater(surface_tag, 0, "Surface tag should be positive.")

        # Verify that one surface was created
        surfaces = gmsh.model.getEntities(2)
        self.assertEqual(len(surfaces), 1, "There should be exactly one surface.")
        self.assertEqual(surfaces[0][1], surface_tag, "Surface tag should match.")

        self.geometry.plot(file_name="test_circle.png")
        self.assertTrue(
            os.path.exists(os.path.join(self.output_dir, "test_circle.png"))
        )

        # For visual debugging, you can uncomment the following line.
        gmsh.fltk.run()
        gmsh.finalize()

    def test_create_triangle(self):
        gmsh.initialize()
        gmsh.model.add("test_triangle")
        surface_tag = self.geometry.triangle((0, 0), (1, 0), (0.5, 1))
        self.assertGreater(surface_tag, 0, "Surface tag should be positive.")

        # Verify that one surface was created
        surfaces = gmsh.model.getEntities(2)
        self.assertEqual(len(surfaces), 1, "There should be exactly one surface.")
        self.assertEqual(surfaces[0][1], surface_tag, "Surface tag should match.")

        self.geometry.plot(file_name="test_triangle.png")
        self.assertTrue(
            os.path.exists(os.path.join(self.output_dir, "test_triangle.png"))
        )

        # For visual debugging, you can uncomment the following line.
        gmsh.fltk.run()
        gmsh.finalize()

    def test_create_ellipse(self):
        gmsh.initialize()
        gmsh.model.add("test_ellipse")
        surface_tag = self.geometry.ellipse(1, 0.5)
        self.assertGreater(surface_tag, 0, "Surface tag should be positive.")

        # Verify that one surface was created
        surfaces = gmsh.model.getEntities(2)
        self.assertEqual(len(surfaces), 1, "There should be exactly one surface.")
        self.assertEqual(surfaces[0][1], surface_tag, "Surface tag should match.")

        self.geometry.plot(file_name="test_ellipse.png")
        self.assertTrue(
            os.path.exists(os.path.join(self.output_dir, "test_ellipse.png"))
        )

        # For visual debugging, you can uncomment the following line.
        gmsh.fltk.run()
        gmsh.finalize()

    def test_create_rectangle_with_partitions(self):
        gmsh.initialize()
        length, width = 5, 2
        mesh_size = 0.2

        gmsh.model.add("test_rectangle_with_partitions")
        surface_tags = self.geometry.rectangle_with_partitions(
            length, width, mesh_size=mesh_size
        )
        self.assertEqual(len(surface_tags), 4, "Should return 4 surface tags.")

        # Verify that four surfaces were created
        surfaces = gmsh.model.occ.getEntities(2)
        self.assertEqual(len(surfaces), 4, "There should be exactly four surfaces.")

        self.geometry.plot(file_name="test_rectangle_with_partitions.png")
        self.assertTrue(
            os.path.exists(
                os.path.join(self.output_dir, "test_rectangle_with_partitions.png")
            )
        )

        # For visual debugging, you can uncomment the following line.
        gmsh.fltk.run()
        gmsh.finalize()


if __name__ == "__main__":
    unittest.main()
