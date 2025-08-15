import unittest
from pathlib import Path
import tempfile

import numpy as np

from src.mesh_analysis import Mesh

from src.mesh_partition import (
    PartitionManager,
    build_halo_indices_from_decomposed,
)


def make_simple_tet_mesh():
    m = Mesh()
    m.dimension = 3
    m.node_coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    m.num_nodes = m.node_coords.shape[0]
    m.cell_connectivity = [[0, 1, 2, 3], [1, 2, 3, 4]]
    m.num_cells = 2
    m.boundary_faces_nodes = np.empty((0, 3), dtype=int)
    m.boundary_faces_tags = np.empty((0,), dtype=int)
    m.analyze_mesh()
    return m


class TestPartitionMesh(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_empty_partitions(self):

        m = make_simple_tet_mesh()
        pm = PartitionManager(m)
        parts = pm.partition_elements(4, method="hierarchical")
        self.assertEqual(parts.shape[0], m.num_cells)
        out = self.tmp_path / "decomp"

        pm.write_decompose_par_json_npy(str(out), 4)
        for p in range(4):
            proc = out / f"processor{p}"
            self.assertTrue(proc.exists())
            self.assertTrue((proc / "mesh.json").exists())

    def test_reconstruction_roundtrip(self):

        m = make_simple_tet_mesh()
        pm = PartitionManager(m)
        pm.partition_elements(2, method="hierarchical")
        out = self.tmp_path / "decomp2"
        pm.write_decompose_par_json_npy(str(out), 2)
        newmesh = pm.reconstruct_par(str(out))
        self.assertEqual(newmesh.num_cells, m.num_cells)

    def test_halo_builder(self):

        m = make_simple_tet_mesh()
        pm = PartitionManager(m)
        pm.partition_elements(2, method="hierarchical")
        out = self.tmp_path / "decomp3"
        pm.write_decompose_par_json_npy(str(out), 2)
        halos = build_halo_indices_from_decomposed(str(out))
        self.assertTrue(set(halos.keys()).issubset({0, 1}))
        for r, info in halos.items():
            self.assertIsInstance(info["neighbors"], dict)

    def test_gmsh_writer(self):

        m = make_simple_tet_mesh()
        pm = PartitionManager(m)
        pm.partition_elements(2, method="hierarchical")
        out = self.tmp_path / "decomp4"
        pm.write_decompose_par_json_npy(str(out), 2)
        pm.write_gmsh_per_processor(str(out), 2)
        for p in range(2):
            self.assertTrue((out / f"processor{p}" / f"processor{p}.msh").exists())


if __name__ == "__main__":
    unittest.main()
