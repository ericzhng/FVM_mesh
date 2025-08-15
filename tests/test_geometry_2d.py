import gmsh
import unittest
import os
from src.create_geometry_2d import Geometry2D

class TestGeometry2D(unittest.TestCase):
    def setUp(self):
        self.output_dir = "d:/2025-08/FVM_mesh/tests/output"
        self.geometry = Geometry2D(output_dir=self.output_dir)
        gmsh.initialize()

    def tearDown(self):
        gmsh.finalize()

    def test_create_polygon(self):
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
        self.geometry.plot_geometry(points, file_name="test_polygon.png")
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "test_polygon.png")))

        gmsh.model.add("test_polygon")

        surface_tag = -1
        try:
            surface_tag = self.geometry.create_polygon(points)
        finally:
            pass

        self.assertTrue(surface_tag > 0)

    def test_create_rectangle(self):
        gmsh.model.add("test_rectangle")

        surface_tag = -1
        try:
            surface_tag = self.geometry.create_rectangle(0, 0, 1, 2)
        finally:
            pass

        self.assertTrue(surface_tag > 0)

    def test_create_circle(self):
        gmsh.model.add("test_circle")

        surface_tag = -1
        try:
            surface_tag = self.geometry.create_circle(0, 0, 1)
        finally:
            pass

        self.assertTrue(surface_tag > 0)

    def test_create_triangle(self):
        gmsh.model.add("test_triangle")

        surface_tag = -1
        try:
            surface_tag = self.geometry.create_triangle((0, 0), (1, 0), (0.5, 1))
        finally:
            pass

        self.assertTrue(surface_tag > 0)

    def test_create_ellipse(self):
        gmsh.model.add("test_ellipse")

        surface_tag = -1
        try:
            surface_tag = self.geometry.create_ellipse(0, 0, 1, 0.5)
        finally:
            pass

        self.assertTrue(surface_tag > 0)

if __name__ == "__main__":
    unittest.main()
