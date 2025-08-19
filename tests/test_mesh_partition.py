import os
import unittest
from pathlib import Path
import tempfile
import numpy as np

os.environ["METIS_DLL"] = os.path.join(os.getcwd(), "dll", "metis.dll")

from src.mesh_analysis import Mesh2D
from src.mesh_partition import (
    partition_mesh,
    reconstruct_mesh_from_decomposed_dir,
    write_decomposed_mesh,
    write_decomposed_mesh_gmsh,
    metis,
)


def make_simple_2d_mesh():
    """Creates a simple 2D mesh with 9 quadrilateral cells for testing."""
    m = Mesh2D()
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
    m.boundary_faces_nodes = np.empty((0, 2), dtype=int)
    m.boundary_faces_tags = np.empty((0,), dtype=int)
    m.analyze_mesh()

    return m


class TestPartitionMesh(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_partition_and_write(self):
        """Test that partitioning runs and writes output files."""
        mesh = make_simple_2d_mesh()
        n_parts = 4
        parts = partition_mesh(mesh, n_parts, method="hierarchical")
        self.assertEqual(parts.shape[0], mesh.num_cells)
        self.assertEqual(len(np.unique(parts)), n_parts)

        output_dir = self.tmp_path / "decomposed"
        write_decomposed_mesh(mesh, parts, str(output_dir), n_parts)

        for p in range(n_parts):
            proc_dir = output_dir / f"processor{p}"
            self.assertTrue(proc_dir.exists())
            self.assertTrue((proc_dir / "mesh.json").exists())
            self.assertTrue((proc_dir / "node_coords.npy").exists())

    def test_reconstruction_roundtrip(self):
        """Test that a decomposed mesh can be reconstructed."""
        mesh = make_simple_2d_mesh()
        n_parts = 4
        parts = partition_mesh(mesh, n_parts, method="hierarchical")

        output_dir = self.tmp_path / "decomposed_roundtrip"
        write_decomposed_mesh(mesh, parts, str(output_dir), n_parts)

        reconstructed_mesh = reconstruct_mesh_from_decomposed_dir(str(output_dir))

        self.assertEqual(reconstructed_mesh.num_cells, mesh.num_cells)
        self.assertEqual(reconstructed_mesh.num_nodes, mesh.num_nodes)
        # Further checks could compare coordinates and connectivity if sorting is guaranteed

    def test_gmsh_writer(self):
        """Test that partitioned gmsh files are written."""
        mesh = make_simple_2d_mesh()
        n_parts = 4
        parts = partition_mesh(mesh, n_parts, method="hierarchical")

        output_dir = self.tmp_path / "decomposed_gmsh"
        write_decomposed_mesh(mesh, parts, str(output_dir), n_parts)
        write_decomposed_mesh_gmsh(str(output_dir), n_parts)

        for p in range(n_parts):
            proc_dir = output_dir / f"processor{p}"
            if (proc_dir / "mesh.json").exists():  # Check if partition is not empty
                self.assertTrue((proc_dir / f"processor{p}.msh").exists())

    @unittest.skipIf(metis is None, "metis is not installed")
    def test_metis_partitioning(self):
        """Test METIS partitioning."""
        mesh = make_simple_2d_mesh()
        n_parts = 4
        parts = partition_mesh(mesh, n_parts, method="metis")
        self.assertEqual(parts.shape[0], mesh.num_cells)
        self.assertEqual(len(np.unique(parts)), n_parts)

    def test_hierarchical_partitioning(self):
        """Test hierarchical partitioning."""
        mesh = make_simple_2d_mesh()
        n_parts = 4
        parts = partition_mesh(mesh, n_parts, method="hierarchical")
        self.assertEqual(parts.shape[0], mesh.num_cells)
        self.assertEqual(len(np.unique(parts)), n_parts)


if __name__ == "__main__":
    unittest.main()
