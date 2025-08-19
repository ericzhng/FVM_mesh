import unittest
from pathlib import Path
import tempfile

import numpy as np

from src.mesh_analysis import Mesh


class TestPartitionMesh(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

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
