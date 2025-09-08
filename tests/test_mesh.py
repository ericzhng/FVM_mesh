import unittest
import tempfile
from pathlib import Path

from src.mesh import Mesh


class TestMeshAnalysis(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)

        self.test_msh_file = "trunk/mixed_mesh_all_surfaces.msh"

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_is_analyzed_flag(self):
        mesh = Mesh()
        self.assertFalse(mesh._is_analyzed)  # Should be False initially
        mesh.read_gmsh(str(self.test_msh_file))
        self.assertFalse(mesh._is_analyzed)  # Should still be False after reading
        mesh.analyze_mesh()
        self.assertTrue(mesh._is_analyzed)  # Should be True after analysis
        mesh._compute_quality()
        mesh.print_summary()
        mesh.plot()


if __name__ == "__main__":
    unittest.main()
