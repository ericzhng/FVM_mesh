import os
import unittest

from src.reorder import renumber_cells, renumber_nodes
from tests.test_partition import make_simple_2d_mesh


class TestRenumber(unittest.TestCase):

    def setUp(self):
        self.tmp_path = "output_reorder"
        os.makedirs(self.tmp_path, exist_ok=True)

    def tearDown(self):
        pass

    def test_renumber(self):
        """Test that renumbering runs and creates new mesh objects."""
        mesh = make_simple_2d_mesh()
        mesh.plot(filepath=os.path.join(self.tmp_path, "original_mesh.png"))

        new_mesh_nodes = renumber_nodes(mesh, strategy="random")
        self.assertNotEqual(id(mesh), id(new_mesh_nodes))
        self.assertEqual(mesh.num_nodes, new_mesh_nodes.num_nodes)
        new_mesh_nodes.plot(
            filepath=os.path.join(self.tmp_path, "mesh_after_renumber_nodes.png")
        )

        new_mesh_cells = renumber_cells(mesh, strategy="sloan")
        self.assertNotEqual(id(mesh), id(new_mesh_cells))
        self.assertEqual(mesh.num_cells, new_mesh_cells.num_cells)
        new_mesh_cells.plot(
            filepath=os.path.join(self.tmp_path, "mesh_after_renumber_cells.png")
        )
