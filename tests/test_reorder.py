import unittest

from src.reorder import renumber_cells, renumber_nodes
from tests.test_partition import make_simple_2d_mesh


class TestRenumber(unittest.TestCase):

    def test_renumber(self):
        """Test that renumbering runs and creates new mesh objects."""
        mesh = make_simple_2d_mesh()
        mesh.plot(file_name="before.png")

        new_mesh_nodes = renumber_nodes(mesh, strategy="reverse")
        self.assertNotEqual(id(mesh), id(new_mesh_nodes))
        self.assertEqual(mesh.num_nodes, new_mesh_nodes.num_nodes)
        new_mesh_nodes.plot(file_name="after_nodes.png")

        new_mesh_cells = renumber_cells(mesh, strategy="sloan")
        self.assertNotEqual(id(mesh), id(new_mesh_cells))
        self.assertEqual(mesh.num_cells, new_mesh_cells.num_cells)
        new_mesh_cells.plot(file_name="after_cells.png")
