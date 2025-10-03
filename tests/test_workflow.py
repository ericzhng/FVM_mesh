
import os
import unittest
import subprocess
from pathlib import Path
import sys
import numpy as np

from polymesh.core_mesh import CoreMesh
from polymesh.polymesh import PolyMesh
from polymesh.distributed_mesh import DistributedMesh
from polymesh.partition import partition_mesh


class TestWorkflow(unittest.TestCase):

    def setUp(self):
        self.tmp_path = "test_output"
        os.makedirs(self.tmp_path, exist_ok=True)
        self.mesh_filepath = os.path.join(os.path.dirname(__file__), "..", "data", "sample_mixed_mesh.msh")

    def tearDown(self):
        # show the temp dir in explorer
        path_to_open = str(Path(self.tmp_path).resolve())
        if sys.platform == "win32":
            subprocess.run(["explorer", path_to_open], check=False)
        elif sys.platform == "darwin":  # macOS
            subprocess.run(["open", path_to_open], check=False)
        else:  # Linux and other UNIX-like
            try:
                subprocess.run(["xdg-open", path_to_open], check=False)
            except FileNotFoundError:
                print(
                    f"Could not open directory {path_to_open}. Please open it manually."
                )

    def test_full_workflow(self):
        """
        Tests the end-to-end workflow:
        1. Generate a global mesh.
        2. Partition the mesh.
        3. For each partition, create a DistributedMesh object.
        4. Verify the properties of the DistributedMesh.
        """
        # 1. Read mesh, analyze and then partition
        core_mesh = CoreMesh()
        core_mesh.read_gmsh(self.mesh_filepath)

        global_mesh = PolyMesh()
        global_mesh.read_gmsh(self.mesh_filepath)
        global_mesh.analyze_mesh()
        global_mesh.print_summary()

        self.assertGreater(
            global_mesh.num_cells, 0, "Mesh should have cells after reading."
        )
        self.assertTrue(global_mesh._is_analyzed, "Mesh should be analyzed.")

        # 2. Partition the mesh into 4 parts
        n_parts = 4
        part_result = partition_mesh(core_mesh, n_parts=n_parts, method="metis")

        self.assertEqual(part_result.n_parts, n_parts)

        local_meshes: list[DistributedMesh] = []
        for i in range(n_parts):
            # 3. Create a DistributedMesh for each partition
            local_mesh = DistributedMesh(global_mesh, part_result, rank=i)
            local_meshes.append(local_mesh)

            # 4. Verify the properties of the DistributedMesh
            self.assertEqual(local_mesh.rank, i)
            self.assertEqual(
                local_mesh.num_cells,
                local_mesh.num_owned_cells + local_mesh.num_halo_cells
            )

            # Check that local cell 0 is the same as the first global owned cell
            self.assertEqual(
                local_mesh.l2g_cells[0], part_result.halo_indices[i]["owned_cells"][0]
            )

            # Check that owned cells are numbered contiguously from 0
            owned_cell_indices_l = list(range(local_mesh.num_owned_cells))
            all_sent_indices = []
            for p in local_mesh.send_map:
                if local_mesh.send_map[p]:
                    all_sent_indices.extend(local_mesh.send_map[p])
            self.assertTrue(np.array_equal(
                owned_cell_indices_l,
                sorted(list(set(all_sent_indices)))
            ))

            # Check that halo cells follow owned cells
            if local_mesh.num_halo_cells > 0:
                first_halo_l = local_mesh.num_owned_cells
                self.assertEqual(
                    local_mesh.l2g_cells[first_halo_l],
                    part_result.halo_indices[i]["halo_cells"][0]
                )

        # Verify send/recv maps
        for i in range(n_parts):
            local_mesh = local_meshes[i]
            for neighbor_rank, send_indices in local_mesh.send_map.items():
                # The other mesh must have a corresponding recv map
                neighbor_mesh = next((m for m in local_meshes if m.rank == neighbor_rank), None)
                if neighbor_mesh:
                    self.assertIn(i, neighbor_mesh.recv_map)
                    self.assertEqual(len(send_indices), len(neighbor_mesh.recv_map[i]))


if __name__ == "__main__":
    unittest.main()
