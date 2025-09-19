import os
import unittest
from pathlib import Path
import tempfile
import numpy as np

os.environ["METIS_DLL"] = os.path.join(os.getcwd(), "dll", "metis.dll")

from src.mesh import Mesh
from src.partition import partition_mesh


def make_simple_2d_mesh():
    """Creates a simple 2D mesh with 9 quadrilateral cells for testing."""
    m = Mesh()
    m.dimension = 2

    # Create a 4x4 grid of nodes
    x = np.linspace(0, 1, 4)
    y = np.linspace(0, 1, 4)
    xx, yy = np.meshgrid(x, y)
    nodes = np.vstack([xx.ravel(), yy.ravel(), np.zeros(xx.size)]).T
    m.node_coords = nodes
    m.num_nodes = m.node_coords.shape[0]

    # Create 3x3 grid of quad cells
    cell_connectivity = []
    for j in range(3):
        for i in range(3):
            # Node indices for the current cell
            n0 = j * 4 + i
            n1 = j * 4 + i + 1
            n2 = (j + 1) * 4 + i + 1
            n3 = (j + 1) * 4 + i
            cell_connectivity.append([n0, n1, n2, n3])

    m.cell_connectivity = cell_connectivity
    m.num_cells = len(cell_connectivity)
    m.analyze_mesh()
    # m.print_summary()
    # m.plot()

    return m


class TestPartitionMesh(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_metis_partitioning(self):
        """Test METIS partitioning."""
        mesh = make_simple_2d_mesh()
        n_parts = 4
        result = partition_mesh(mesh, n_parts, method="metis")
        result.print_summary()
        self.assertEqual(result.parts.shape[0], mesh.num_cells)
        self.assertEqual(len(np.unique(result.parts)), n_parts)
        mesh.plot(file_name="test_partition_metis.png", parts=result.parts)

    def test_hierarchical_partitioning(self):
        """Test hierarchical partitioning."""
        mesh = make_simple_2d_mesh()
        n_parts = 4
        result = partition_mesh(mesh, n_parts, method="hierarchical")
        result.print_summary()
        self.assertEqual(result.parts.shape[0], mesh.num_cells)
        self.assertEqual(len(np.unique(result.parts)), n_parts)
        mesh.plot(file_name="test_partition.png", parts=result.parts)

    def test_halo_indices(self):
        """Test that halo indices are created correctly."""
        mesh = make_simple_2d_mesh()
        n_parts = 4
        result = partition_mesh(mesh, n_parts, method="metis")
        mesh.plot(file_name="test_partition_metis_check.png", parts=result.parts)

        halo_indices = result.halo_indices
        self.assertIsInstance(halo_indices, dict)
        self.assertEqual(len(halo_indices), n_parts)

        for rank in range(n_parts):
            self.assertIn(rank, halo_indices)
            self.assertIn("owned_cells", halo_indices[rank])
            self.assertIn("send", halo_indices[rank])
            self.assertIn("recv", halo_indices[rank])


if __name__ == "__main__":
    unittest.main()
