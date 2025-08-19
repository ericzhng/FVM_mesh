import os
import unittest
from pathlib import Path
import tempfile
import numpy as np

from src.utility import print_partition

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
    m.print_summary()
    m.plot()

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
        parts = partition_mesh(mesh, n_parts, method="metis")
        print_partition(parts)
        self.assertEqual(parts.shape[0], mesh.num_cells)
        self.assertEqual(len(np.unique(parts)), n_parts)
        mesh.plot(file_name="test_partition.png", parts=parts)

    def test_hierarchical_partitioning(self):
        """Test hierarchical partitioning."""
        mesh = make_simple_2d_mesh()
        n_parts = 4
        parts = partition_mesh(mesh, n_parts, method="hierarchical")
        print_partition(parts)
        self.assertEqual(parts.shape[0], mesh.num_cells)
        self.assertEqual(len(np.unique(parts)), n_parts)
        mesh.plot(file_name="test_partition.png", parts=parts)


if __name__ == "__main__":
    unittest.main()
