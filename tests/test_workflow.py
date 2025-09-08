import os
import unittest
from pathlib import Path
import subprocess

import gmsh

from src.geometry import Geometry
from src.mesh import Mesh
from src.mesh_generator import MeshGenerator
from src.partition import partition_mesh


class TestWorkflow(unittest.TestCase):

    def setUp(self):
        self.tmp_path = "./test_output"
        os.makedirs(self.tmp_path, exist_ok=True)

    def tearDown(self):
        # show the temp dir in explorer
        subprocess.run(["explorer", str(Path(self.tmp_path).resolve())])

    def test_meshing_and_partition_workflow(self):
        """An integration test for the full workflow from geometry to partitioned mesh."""
        # 1. Create geometry and then mesh
        gmsh.initialize()
        gmsh.model.add("test_rect_workflow")
        geom = Geometry(output_dir=self.tmp_path)
        surfaces = geom.rectangle(length=2, width=1, mesh_size=0.5)

        mesh_filename = "test_mesh.msh"
        mesh_filepath = Path(self.tmp_path) / mesh_filename
        mesh_gen = MeshGenerator(surface_tags=surfaces, output_dir=self.tmp_path)
        mesh_params = {surfaces: {"mesh_type": "tri", "char_length": 0.5}}
        mesh_gen.generate(mesh_params=mesh_params, filename=mesh_filename)
        gmsh.finalize()

        self.assertTrue(mesh_filepath.exists(), "Mesh file was not created.")

        # 2. Read mesh, analyze and then partition
        mesh = Mesh()
        mesh.read_gmsh(str(mesh_filepath))
        mesh.analyze_mesh()
        mesh.print_summary()

        self.assertGreater(mesh.num_cells, 0, "Mesh should have cells after reading.")
        self.assertTrue(mesh._is_analyzed, "Mesh should be analyzed.")

        # 3. Partition the mesh
        n_parts = 3
        result = partition_mesh(mesh, n_parts, method="hierarchical")
        result.print_summary()

        self.assertEqual(result.n_parts, n_parts)
        self.assertEqual(result.parts.shape[0], mesh.num_cells)

        plot_filename = "mesh_plot_partition.png"
        mesh.plot(
            file_name=str(Path(self.tmp_path) / plot_filename), parts=result.parts
        )

        # 4. Check halo indices
        halo_indices = result.halo_indices
        self.assertIsInstance(halo_indices, dict)
        self.assertEqual(len(halo_indices), n_parts)


if __name__ == "__main__":
    unittest.main()
