import os
import unittest
import subprocess
from pathlib import Path
import sys
import numpy as np
from typing import List

from polymesh.poly_mesh import PolyMesh
from polymesh.local_mesh import LocalMesh, _compute_halo_indices, create_local_meshes
from polymesh.partition import partition_mesh, print_partition_summary
from tests.common_meshes import create_5x5_quad_mesh_fixture


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

    def _verify_comm_maps(self, local_meshes: List[LocalMesh]):
        """Helper function to verify send/recv maps between local meshes."""
        n_parts = len(local_meshes)
        for i in range(n_parts):
            local_mesh = local_meshes[i]
            for neighbor_rank, send_indices in local_mesh.send_map.items():
                neighbor_mesh = next(
                    (m for m in local_meshes if m.rank == neighbor_rank), None
                )

                if neighbor_mesh is None:
                    self.fail(
                        f"Neighbor mesh with rank {neighbor_rank} not found among local_meshes"
                    )

                # Check for corresponding recv map
                self.assertIn(i, neighbor_mesh.recv_map)
                recv_indices = neighbor_mesh.recv_map[i]
                self.assertEqual(len(send_indices), len(recv_indices))

                # Check that the global cells are the same
                global_sent_cells = local_mesh.l2g_cells[send_indices]
                global_recv_cells = neighbor_mesh.l2g_cells[recv_indices]

                # Using assertCountEqual for order-independent comparison
                self.assertCountEqual(
                    global_sent_cells.tolist(), global_recv_cells.tolist()
                )

    def test_reordering_options(self):
        """
        Tests the reordering capabilities of the LocalMesh class.
        """
        global_mesh = PolyMesh.from_gmsh(self.mesh_filepath)
        global_mesh.analyze_mesh()

        n_parts = 2
        # 1. Test with cell and node reordering enabled
        local_meshes = create_local_meshes(
            global_mesh,
            n_parts=n_parts,
            partition_method="metis",
            reorder_cells_strategy="rcm",
            reorder_nodes_strategy="rcm",
        )
        self.assertEqual(len(local_meshes), n_parts)
        local_mesh = local_meshes[0]
        self.assertTrue(local_mesh.use_reordered_cells)
        self.assertTrue(local_mesh.use_reordered_nodes)

        # Store a copy of the reordered maps for comparison
        reordered_l2g_cells = local_mesh.l2g_cells.copy()
        reordered_l2g_nodes = local_mesh.l2g_nodes.copy()

        # 2. Test restoring original cell ordering
        local_mesh.reorder_cells(active=False)
        self.assertFalse(local_mesh.use_reordered_cells)
        # The l2g map should be different now
        self.assertFalse(np.array_equal(reordered_l2g_cells, local_mesh.l2g_cells))

        # 3. Test restoring original node ordering
        local_mesh.reorder_nodes(active=False)
        self.assertFalse(local_mesh.use_reordered_nodes)
        self.assertFalse(np.array_equal(reordered_l2g_nodes, local_mesh.l2g_nodes))

        # 4. Test creating meshes without reordering
        local_meshes_no_reorder = create_local_meshes(
            global_mesh,
            n_parts=n_parts,
            partition_method="metis",
            reorder_cells_strategy=None,
        )
        self.assertFalse(local_meshes_no_reorder[0].use_reordered_cells)

    def test_5x5_mesh_workflow(self):
        """
        Tests the workflow with a simple 5x5 structured quad mesh for debugging.
        """
        global_mesh, parts, n_parts = create_5x5_quad_mesh_fixture()
        global_mesh.plot(filepath=f"{self.tmp_path}/5x5_global_mesh.png", parts=parts)

        # Create local meshes using the predefined partition and reorder them
        local_meshes = create_local_meshes(
            global_mesh,
            parts=parts,
            reorder_cells_strategy="",
            reorder_nodes_strategy="",
        )
        self.assertEqual(len(local_meshes), n_parts)

        for i in range(n_parts):
            local_mesh = local_meshes[i]
            local_mesh.plot(
                filepath=f"{self.tmp_path}/5x5_local_mesh_rank_{local_mesh.rank}.png"
            )
            local_mesh.print_summary()

        # Verify communication maps
        self._verify_comm_maps(local_meshes)

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

        # In test_full_workflow, we now test create_local_meshes with n_parts
        local_meshes: list[LocalMesh] = create_local_meshes(
            global_mesh,
            n_parts=n_parts,
            partition_method="metis",
            reorder_cells_strategy="",
            reorder_nodes_strategy="",
        )

        self.assertEqual(len(local_meshes), n_parts)

        # Plot the global mesh with partition info from one of the local meshes
        # This requires reconstructing the global parts array
        parts = np.zeros(global_mesh.num_cells, dtype=int)
        for lm in local_meshes:
            for i, g_cell in enumerate(lm.l2g_cells):
                if i < lm.num_owned_cells:
                    parts[g_cell] = lm.rank
        global_mesh.plot(filepath=f"{self.tmp_path}/global_mesh.png", parts=parts)

        for i in range(n_parts):
            local_mesh = local_meshes[i]

            local_mesh.plot(
                filepath=f"{self.tmp_path}/local_mesh_rank_{local_mesh.rank}.png"
            )
            local_mesh.print_summary()

        # Verify send/recv maps
        self._verify_comm_maps(local_meshes)


if __name__ == "__main__":
    unittest.main()
