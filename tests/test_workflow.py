import os
import unittest
import subprocess
from pathlib import Path
import sys
import numpy as np

from polymesh.poly_mesh import PolyMesh
from polymesh.local_mesh import LocalMesh, create_local_meshes
from polymesh.partition import partition_mesh


class TestWorkflow(unittest.TestCase):

    def setUp(self):
        self.tmp_path = "results/workflow"
        os.makedirs(self.tmp_path, exist_ok=True)
        self.mesh_filepath = os.path.join(
            os.path.dirname(__file__), "..", "data", "sample_mixed_mesh.msh"
        )

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
        3. For each partition, create a LocalMesh object.
        4. Verify the properties of the LocalMesh.
        """
        # 1. Read mesh, analyze and then partition
        global_mesh = PolyMesh.from_gmsh(self.mesh_filepath)
        global_mesh.analyze_mesh()
        global_mesh.print_summary()

        self.assertGreater(
            global_mesh.num_cells, 0, "Mesh should have cells after reading."
        )
        self.assertTrue(global_mesh._is_analyzed, "Mesh should be analyzed.")

        # 2. Partition the mesh into 4 parts
        n_parts = 4
        parts = partition_mesh(global_mesh, n_parts=n_parts, method="metis")

        global_mesh.plot(filepath=f"{self.tmp_path}/global_mesh.png", parts=parts)

        # get number of parts from partition result
        self.assertEqual(len(np.unique(parts)), n_parts)

        local_meshes: list[LocalMesh] = create_local_meshes(
            global_mesh, n_parts, "metis"
        )

        self.assertEqual(len(local_meshes), n_parts)

        for i in range(n_parts):
            local_mesh = local_meshes[i]

            local_mesh.plot(
                filepath=f"{self.tmp_path}/local_mesh_rank_{local_mesh.rank}.png"
            )
            local_mesh.print_summary()

            # 4. Verify the properties of the LocalMesh
            self.assertEqual(local_mesh.rank, i)
            self.assertEqual(
                local_mesh.num_cells,
                local_mesh.num_owned_cells + local_mesh.num_halo_cells,
            )

        # Verify send/recv maps
        for i in range(n_parts):
            local_mesh = local_meshes[i]
            for neighbor_rank, send_indices in local_mesh.send_map.items():
                # The other mesh must have a corresponding recv map
                neighbor_mesh = next(
                    (m for m in local_meshes if m.rank == neighbor_rank), None
                )
                if neighbor_mesh:
                    self.assertIn(i, neighbor_mesh.recv_map)
                    self.assertEqual(len(send_indices), len(neighbor_mesh.recv_map[i]))


if __name__ == "__main__":
    unittest.main()
