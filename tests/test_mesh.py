import unittest
import tempfile
from pathlib import Path

from polymesh.mesh import Mesh


class TestMeshAnalysis(unittest.TestCase):

    def setUp(self):
        self.tmp_path = "output_generation"
        self.test_msh_file = "data/mixed_mesh_all_surfaces.msh"

    def tearDown(self):
        pass

    def test_mesh_analysis(self):
        mesh = Mesh()
        mesh.read_gmsh(str(self.test_msh_file))
        mesh.analyze_mesh()
        self.assertTrue(mesh._is_analyzed)  # Should be True after analysis

        mesh.print_summary()
        mesh.plot(filepath=self.tmp_path + "/mesh_plot.png")


if __name__ == "__main__":
    unittest.main()
