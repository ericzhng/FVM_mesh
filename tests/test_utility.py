import unittest
from pathlib import Path
import tempfile

import numpy as np

from src.mesh import Mesh
from src.partition import partition_mesh
from src.utility import (
    build_halo_indices,
    print_partition,
    renumber_cells,
    renumber_nodes,
)
from tests.test_partition import make_simple_2d_mesh


class TestPartitionMesh(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_halo_builder(self):
        pass

    def test_partition_halo(self):
        """Test that partitioning runs and writes output files."""
        mesh = make_simple_2d_mesh()
        n_parts = 4
        parts = partition_mesh(mesh, n_parts, method="metis")
        print_partition(parts)
        mesh.plot(file_name="test_partition.png", parts=parts)

        out = build_halo_indices(mesh, parts)
        print(out)

    def test_renumber(self):
        """Test that partitioning runs and writes output files."""
        mesh = make_simple_2d_mesh()
        mesh.plot(file_name="before.png")

        new_mesh = renumber_nodes(mesh, strategy="reverse")
        new_mesh.plot(file_name="after_nodes.png")

        new_mesh = renumber_cells(mesh, strategy="sloan")
        new_mesh.plot(file_name="after_cells.png")
