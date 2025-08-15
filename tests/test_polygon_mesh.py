import os
import sys
import gmsh

import unittest
from pathlib import Path
import tempfile

# Add the src directory to the python path to import the function
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.create_geometry_2d import create_polygon_from_points
from src.create_mesh_2d import create_mesh_from_geometry


class TestCreateMesh(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)

    # Example usage for a triangular mesh from a set of points
    points = [
        (0, 0),
        (1, 0),
        (1, 1),
        (0, 1),
        (0.5, 0.5),
        (0.2, 0.8),
        (0.8, 0.2),
        (0.5, 0.1),
    ]

    gmsh.initialize()
    gmsh.model.add("polygon_example")

    try:
        surface_tag = create_polygon_from_points(points)
        print(f"Successfully created surface with tag: {surface_tag}")

        # Mesh the geometry
        create_mesh_from_geometry(
            surface_tag,
            filename="d:/2025-08/FVM_mesh/trunk/polygon_triangular.msh",
            mesh_type="triangular",
            char_length=0.1,
        )

    except ValueError as e:
        print(f"Error: {e}")
    finally:
        if gmsh.isInitialized():
            gmsh.finalize()

    # Example for a structured mesh
    points_rect = [(0, 0), (2, 0), (2, 1), (0, 1)]

    gmsh.initialize()
    gmsh.model.add("rectangle_example")

    try:
        surface_tag_rect = create_polygon_from_points(points_rect)
        print(f"Successfully created surface with tag: {surface_tag_rect}")

        # Mesh the geometry
        create_mesh_from_geometry(
            surface_tag_rect,
            filename="d:/2025-08/FVM_mesh/trunk/polygon_structured.msh",
            mesh_type="structured",
            nx=20,
            ny=10,
        )

    except ValueError as e:
        print(f"Error: {e}")
    finally:
        if gmsh.isInitialized():
            gmsh.finalize()


if __name__ == "__main__":
    unittest.main()
