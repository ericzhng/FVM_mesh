import unittest
from pathlib import Path
import tempfile

import numpy as np

from src.mesh import Mesh2D


class TestPartitionMesh(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_halo_builder(self):
        pass

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
