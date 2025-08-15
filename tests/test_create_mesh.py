import unittest
from pathlib import Path
import tempfile

import numpy as np
from src.mesh_analysis import Mesh


class TestCreateMesh(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)

        # Define rectangle dimensions and the number of elements
        rect_length = 100.0
        rect_height = 100.0
        num_elements_x = 8
        num_elements_y = 8

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_structured(self):

        # --- Create Structured Mesh ---
        print(
            f"Creating a structured mesh of size {rect_length}x{rect_height} "
            f"with {num_elements_x}x{num_elements_y} elements."
        )
        create_and_mesh_rectangle(
            rect_length,
            rect_height,
            num_elements_x,
            num_elements_y,
            filename="data/river_structured.msh",
            mesh_type="structured",
        )
        print("\n" + "=" * 40 + "\n")

    def test_triangular(self):
        # --- Create Triangular Mesh ---
        print(
            f"Creating a triangular mesh of size {rect_length}x{rect_height} "
            f"with approximately {num_elements_x}x{num_elements_y}x2 element resolution."
        )
        # For triangular mesh, nx and ny control the mesh density at the boundaries
        create_and_mesh_rectangle(
            rect_length,
            rect_height,
            num_elements_x,
            num_elements_y,
            filename="data/river_triangular.msh",
            mesh_type="triangular",
        )
        print("\n" + "=" * 40 + "\n")

    def test_mixed(self):
        # --- Create Mixed Mesh ---
        print(
            f"Creating a mixed mesh of size {rect_length}x{rect_height} "
            f"with a structured left side and triangular right side."
        )
        create_and_mesh_rectangle(
            rect_length,
            rect_height,
            num_elements_x,
            num_elements_y,
            filename="data/river_mixed.msh",
            mesh_type="mixed",
        )

        print("\nScript finished.")

    def test_gmsh_creator(self):
        # Example run: read mesh, analyze, partition and write processor folders
        msh_path = "./data/river_mixed.msh"
        out_dir = "./decomposed"
        nprocs = 4

        mesh = Mesh()
        mesh.read_gmsh(msh_path)
        mesh.analyze_mesh()
        mesh_summary = mesh.get_mesh_data()
        print("Loaded mesh with", mesh.num_nodes, "nodes and", mesh.num_cells, "cells")

        pm = PartitionManager(mesh)
        parts = pm.partition_elements(nprocs)
        print("Partition counts:", np.bincount(parts))

        pm.write_decompose_par(out_dir, nprocs)
        print(f"Wrote decomposition into {out_dir}/processor* directories")


if __name__ == "__main__":
    unittest.main()
