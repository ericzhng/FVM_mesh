
import unittest
import os

from polymesh.polymesh import PolyMesh


class TestPolyMesh(unittest.TestCase):

    def setUp(self):
        self.tmp_path = "output_generation"
        self.test_msh_file = os.path.join(os.path.dirname(__file__), "..", "data", "sample_mixed_mesh.msh")

    def test_mesh_analysis(self):
        mesh = PolyMesh(self.test_msh_file)
        mesh.analyze_mesh()
        self.assertTrue(mesh._is_analyzed)  # Should be True after analysis

        mesh.print_summary()
        mesh.plot(filepath=os.path.join(self.tmp_path, "mesh_plot.png"))


if __name__ == "__main__":
    unittest.main()
