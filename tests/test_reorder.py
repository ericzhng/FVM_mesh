import os
import unittest
import numpy as np
import copy

from polymesh.reorder import renumber_cells, renumber_nodes
from polymesh.core_mesh import CoreMesh
from polymesh.local_mesh import LocalMesh
from tests.common_meshes import create_2x2_quad_mesh_fixture


def make_test_mesh_from_file():
    """Creates a test mesh from a file."""
    msh_file = os.path.join(
        os.path.dirname(__file__), "..", "data", "sample_mixed_mesh.msh"
    )
    mesh = CoreMesh()
    mesh.read_gmsh(msh_file)
    return mesh


class TestRenumberSmoke(unittest.TestCase):
    """
    Basic smoke tests to ensure renumbering functions run without crashing.
    """

    def setUp(self):
        self.tmp_path = "results/reorder"
        os.makedirs(self.tmp_path, exist_ok=True)

    def test_renumber_smoke(self):
        """Test that renumbering runs and modifies the mesh in-place."""
        mesh = make_test_mesh_from_file()
        mesh.plot(filepath=f"{self.tmp_path}/original_mesh.png")
        mesh_copy = copy.deepcopy(mesh)

        result_nodes = renumber_nodes(mesh_copy, strategy="random")
        self.assertIsNone(result_nodes)
        self.assertEqual(mesh.num_nodes, mesh_copy.num_nodes)
        mesh_copy.plot(filepath=f"{self.tmp_path}/renumbered_nodes.png")

        mesh_copy = copy.deepcopy(mesh)
        result_cells = renumber_cells(mesh_copy, strategy="sloan")
        self.assertIsNone(result_cells)
        self.assertEqual(mesh.num_cells, mesh_copy.num_cells)
        mesh_copy.plot(filepath=f"{self.tmp_path}/renumbered_cells.png")


class TestRenumberCells(unittest.TestCase):
    """
    Tests for the renumber_cells function with various strategies.
    """

    def setUp(self):
        self.mesh, _, _ = create_2x2_quad_mesh_fixture()
        self.tmp_path = "results/reorder"
        os.makedirs(self.tmp_path, exist_ok=True)

    def test_renumber_cells_preserves_mesh(self):
        """Check that reordering produces a valid permutation of the original cells."""
        for strategy in ["rcm", "gps", "sloan", "spectral", "spatial_x", "random"]:
            with self.subTest(strategy=strategy):
                new_mesh = copy.deepcopy(self.mesh)
                renumber_cells(new_mesh, strategy=strategy)
                self.assertEqual(self.mesh.num_cells, new_mesh.num_cells)
                new_mesh.plot(
                    filepath=f"{self.tmp_path}/renumbered_cells_{strategy}.png"
                )

                # Check that the set of cells is the same, just in a different order
                original_conn_sets = {
                    tuple(sorted(c)) for c in self.mesh.cell_connectivity
                }
                new_conn_sets = {tuple(sorted(c)) for c in new_mesh.cell_connectivity}
                self.assertEqual(original_conn_sets, new_conn_sets)

    def test_renumber_cells_spatial_x(self):
        """Test cell reordering with the predictable spatial_x strategy."""
        # Centroids of the simple mesh are at x=0.5, 1.5, 0.5, 1.5
        # Expected order should group cells by x-coordinate of centroid.
        # Original cells 0 and 2 have x=0.5. Original cells 1 and 3 have x=1.5.
        new_mesh = copy.deepcopy(self.mesh)
        renumber_cells(new_mesh, strategy="spatial_x")

        # The new connectivity should be a permutation of the original
        # The first two cells should be the original cells 0 and 2 (in any order)
        first_two_cells = {tuple(sorted(c)) for c in new_mesh.cell_connectivity[:2]}
        self.assertEqual(
            first_two_cells,
            {
                tuple(sorted(self.mesh.cell_connectivity[0])),
                tuple(sorted(self.mesh.cell_connectivity[2])),
            },
        )

        # The last two cells should be the original cells 1 and 3 (in any order)
        last_two_cells = {tuple(sorted(c)) for c in new_mesh.cell_connectivity[2:]}
        self.assertEqual(
            last_two_cells,
            {
                tuple(sorted(self.mesh.cell_connectivity[1])),
                tuple(sorted(self.mesh.cell_connectivity[3])),
            },
        )


class TestRenumberNodes(unittest.TestCase):
    """
    Tests for the renumber_nodes function.
    """

    def setUp(self):
        self.mesh, _, _ = create_2x2_quad_mesh_fixture()

    def test_renumber_nodes_preserves_geometry(self):
        """Check that reordering nodes preserves the geometric integrity of cells."""
        for strategy in ["rcm", "sequential", "reverse", "spatial_x", "random"]:
            with self.subTest(strategy=strategy):
                new_mesh = copy.deepcopy(self.mesh)
                renumber_nodes(new_mesh, strategy=strategy)
                self.assertEqual(self.mesh.num_nodes, new_mesh.num_nodes)
                new_mesh.plot(
                    filepath=f"results/reorder/renumbered_nodes_{strategy}.png"
                )

                # Check that the set of node coordinates for each cell is preserved
                for new_cell_conn in new_mesh.cell_connectivity:
                    new_nodes_coords = new_mesh.node_coords[new_cell_conn]

                    # Find a matching cell in the original mesh by comparing node coordinates
                    found_match = any(
                        np.allclose(
                            np.sort(new_nodes_coords, axis=0),
                            np.sort(self.mesh.node_coords[old_cell_conn], axis=0),
                        )
                        for old_cell_conn in self.mesh.cell_connectivity
                    )
                    self.assertTrue(
                        found_match,
                        f"Cell geometry not preserved with strategy '{strategy}'",
                    )

    def test_renumber_nodes_spatial_x(self):
        """Test node reordering with the predictable spatial_x strategy."""
        new_mesh = copy.deepcopy(self.mesh)
        renumber_nodes(new_mesh, strategy="spatial_x")

        # Expected order of nodes based on x-coordinate:
        # Nodes 0,3,6 have x=0. Nodes 1,4,7 have x=1. Nodes 2,5,8 have x=2.
        expected_x_coords = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
        np.testing.assert_array_equal(new_mesh.node_coords[:, 0], expected_x_coords)


class TestRenumberLocalMesh(unittest.TestCase):
    """
    Tests renumbering on a LocalMesh object.
    """

    def setUp(self):
        self.global_mesh, self.parts, _ = create_2x2_quad_mesh_fixture()
        self.rank = 0
        self.local_mesh = LocalMesh(self.global_mesh, self.parts, self.rank)
        self.tmp_path = "results/reorder"
        os.makedirs(self.tmp_path, exist_ok=True)

    def test_renumber_owned_cells(self):
        """
        Check that only owned cells are reordered in a LocalMesh.
        """
        local_mesh_copy = copy.deepcopy(self.local_mesh)

        # Get the original halo cells to compare later
        original_halo_connectivity = local_mesh_copy.cell_connectivity[
            local_mesh_copy.num_owned_cells :
        ]

        renumber_cells(local_mesh_copy, strategy="random")

        self.assertEqual(self.local_mesh.num_cells, local_mesh_copy.num_cells)
        self.assertEqual(
            self.local_mesh.num_owned_cells, local_mesh_copy.num_owned_cells
        )
        self.assertEqual(self.local_mesh.num_halo_cells, local_mesh_copy.num_halo_cells)

        # Check that halo cells are untouched
        new_halo_connectivity = local_mesh_copy.cell_connectivity[
            local_mesh_copy.num_owned_cells :
        ]
        self.assertEqual(original_halo_connectivity, new_halo_connectivity)

        # Check that owned cells are a permutation of the original owned cells
        original_owned_conn_sets = {
            tuple(sorted(c))
            for c in self.local_mesh.cell_connectivity[
                : self.local_mesh.num_owned_cells
            ]
        }
        new_owned_conn_sets = {
            tuple(sorted(c))
            for c in local_mesh_copy.cell_connectivity[
                : local_mesh_copy.num_owned_cells
            ]
        }
        self.assertEqual(original_owned_conn_sets, new_owned_conn_sets)


if __name__ == "__main__":
    unittest.main()
