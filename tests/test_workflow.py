import os
import unittest
import subprocess
from pathlib import Path

import gmsh
import numpy as np

from src.geometry import Geometry
from src.mesh_generator import MeshGenerator
from src.mesh import Mesh
from src.local_mesh import LocalMesh
from src.partition import partition_mesh


class TestWorkflow(unittest.TestCase):

    def setUp(self):
        self.tmp_path = "./test_output"
        os.makedirs(self.tmp_path, exist_ok=True)

        mesh_filename = "test_mesh.msh"
        mesh_filepath = Path(self.tmp_path) / mesh_filename

        # Create geometry and then mesh
        gmsh.initialize()
        gmsh.model.add("rect_grid_mesh")
        geom = Geometry(output_dir=self.tmp_path)
        surfaces = geom.rectangle(length=10, width=10, mesh_size=1)

        mesh_gen = MeshGenerator(surface_tags=surfaces, output_dir=self.tmp_path)
        mesh_params = {surfaces: {"mesh_type": "tri", "char_length": 0.5}}
        mesh_gen.generate(mesh_params=mesh_params, filename=mesh_filename)
        gmsh.finalize()

        self.assertTrue(mesh_filepath.exists(), "Mesh file was not created.")

    def tearDown(self):
        # show the temp dir in explorer
        subprocess.run(["explorer", str(Path(self.tmp_path).resolve())])

    def test_full_workflow(self):
        """
        Tests the end-to-end workflow:
        1. Generate a global mesh.
        2. Partition the mesh.
        3. For each partition, create a LocalMesh object.
        4. Verify the properties of the LocalMesh.
        """
        # 1. Generate a global mesh (e.g., a 10x10 grid)

        # 2. Read mesh, analyze and then partition
        mesh = Mesh()
        mesh.read_gmsh(str(mesh_filepath))
        mesh.analyze_mesh()
        mesh.print_summary()

        self.assertGreater(mesh.num_cells, 0, "Mesh should have cells after reading.")
        self.assertTrue(mesh._is_analyzed, "Mesh should be analyzed.")

        # 2. Partition the mesh into 4 parts
        n_parts = 4
        part_result = partition_mesh(global_mesh, n_parts=n_parts, method="metis")

        assert part_result.n_parts == n_parts

        local_meshes: list[LocalMesh] = []
        for i in range(n_parts):
            # 3. Create a LocalMesh for each partition
            local_mesh = LocalMesh(global_mesh, part_result, rank=i)
            local_meshes.append(local_mesh)

            # 4. Verify the properties of the LocalMesh
            assert local_mesh.rank == i
            assert (
                local_mesh.mesh.num_cells
                == local_mesh.num_owned_cells + local_mesh.num_halo_cells
            )

            # Check that local cell 0 is the same as the first global owned cell
            assert (
                local_mesh.l2g_cells[0] == part_result.halo_indices[i]["owned_cells"][0]
            )

            # Check that owned cells are numbered contiguously from 0
            owned_cell_indices_l = list(range(local_mesh.num_owned_cells))
            assert np.array_equal(
                owned_cell_indices_l,
                sorted(
                    list(local_mesh.send_map[p])[0]
                    for p in local_mesh.send_map
                    if local_mesh.send_map[p]
                ),
            )

            # Check that halo cells follow owned cells
            if local_mesh.num_halo_cells > 0:
                first_halo_l = local_mesh.num_owned_cells
                assert (
                    local_mesh.l2g_cells[first_halo_l]
                    == part_result.halo_indices[i]["halo_cells"][0]
                )

            # Verify send/recv maps
            for neighbor_rank, send_indices in local_mesh.send_map.items():
                # The other mesh must have a corresponding recv map
                neighbor_mesh = next(m for m in local_meshes if m.rank == neighbor_rank)
                assert i in neighbor_mesh.recv_map
                assert len(send_indices) == len(neighbor_mesh.recv_map[i])


if __name__ == "__main__":
    unittest.main()
