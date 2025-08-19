import unittest
from pathlib import Path
import tempfile

import numpy as np

from src.mesh_analysis import Mesh2D


class TestPartitionMesh(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_halo_builder(self):
        pass
