import unittest
import tempfile
from pathlib import Path

from src.mesh import Mesh


class TestMeshAnalysis(unittest.TestCase):

    def setUp(self):
        self.tmp_path = "output_generation"
        self.test_msh_file = "data/mixed_mesh_all_surfaces.msh"

    def tearDown(self):
        pass

    def test_is_analyzed_flag(self):
        mesh = Mesh()
        self.assertFalse(mesh._is_analyzed)  # Should be False initially
        mesh.read_gmsh(str(self.test_msh_file))
        self.assertFalse(mesh._is_analyzed)  # Should still be False after reading
        mesh.analyze_mesh()
        self.assertTrue(mesh._is_analyzed)  # Should be True after analysis
        mesh._compute_quality()
        mesh.print_summary()
        mesh.plot(plot_file=self.tmp_path + "/mesh_plot.png")


if __name__ == "__main__":
    unittest.main()
