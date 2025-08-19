import unittest
import tempfile
from pathlib import Path

import gmsh
from src.mesh_analysis import Mesh2D


# Helper function to create a simple 2D mesh for testing
def create_test_2d_mesh(filename="test_mesh.msh"):
    gmsh.initialize()
    gmsh.model.add("test_model")
    # Create a simple rectangle
    lc = 1e-1  # Characteristic length
    p1 = gmsh.model.geo.addPoint(0, 0, 0, lc)
    p2 = gmsh.model.geo.addPoint(1, 0, 0, lc)
    p3 = gmsh.model.geo.addPoint(1, 1, 0, lc)
    p4 = gmsh.model.geo.addPoint(0, 1, 0, lc)
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)
    cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    s = gmsh.model.geo.addPlaneSurface([cl])
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)  # Generate 2D mesh
    gmsh.write(filename)
    gmsh.finalize()


class TestMeshAnalysis(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)
        # self.test_msh_file = self.tmp_path / "test_mesh.msh"
        # create_test_2d_mesh(str(self.test_msh_file))  # Create a test mesh for each test

        self.test_msh_file = "test_output/mixed_mesh_all_surfaces.msh"

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_is_analyzed_flag(self):
        mesh = Mesh2D()
        self.assertFalse(mesh._is_analyzed)  # Should be False initially
        mesh.read_gmsh(str(self.test_msh_file))
        self.assertFalse(mesh._is_analyzed)  # Should still be False after reading
        mesh.analyze_mesh()
        self.assertTrue(mesh._is_analyzed)  # Should be True after analysis
        mesh.compute_quality()
        mesh.print_summary()


if __name__ == "__main__":
    unittest.main()
