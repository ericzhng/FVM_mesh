import gmsh
import unittest

# Add the src directory to the python path to import the function
# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.create_geometry_2d import create_polygon, plot_geometry


class TestCreateGeometry(unittest.TestCase):
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
        plot_geometry(points)

        gmsh.initialize()
        gmsh.model.add("test_polygon")

        surface_tag = -1
        try:
            surface_tag = create_polygon(points)
        finally:
            gmsh.finalize()

        self.assertTrue(surface_tag > 0)


if __name__ == "__main__":
    unittest.main()
