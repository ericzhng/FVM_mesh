import os
import unittest

from polymesh.reorder import renumber_cells, renumber_nodes
from polymesh.core_mesh import CoreMesh

def make_test_mesh():
    """Creates a test mesh from a file."""
    msh_file = os.path.join(os.path.dirname(__file__), "..", "data", "sample_mixed_mesh.msh")
    mesh = CoreMesh()
    mesh.read_gmsh(msh_file)
    return mesh

class TestRenumber(unittest.TestCase):

    def setUp(self):
        self.tmp_path = "output_reorder"
        os.makedirs(self.tmp_path, exist_ok=True)

    def test_renumber(self):
        """Test that renumbering runs and creates new mesh objects."""
        mesh = make_test_mesh()

        new_mesh_nodes = renumber_nodes(mesh, strategy="random")
        self.assertNotEqual(id(mesh), id(new_mesh_nodes))
        self.assertEqual(mesh.num_nodes, new_mesh_nodes.num_nodes)

        new_mesh_cells = renumber_cells(mesh, strategy="sloan")
        self.assertNotEqual(id(mesh), id(new_mesh_cells))
        self.assertEqual(mesh.num_cells, new_mesh_cells.num_cells)


if __name__ == "__main__":
    unittest.main()