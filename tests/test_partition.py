import os
import unittest

import gmsh
import numpy as np

from meshgen.geometry import Geometry
from meshgen.mesh_generator import MeshGenerator
from polymesh.mesh import Mesh

os.environ["METIS_DLL"] = os.path.join(os.getcwd(), "dll", "metis.dll")
from polymesh.partition import partition_mesh


def make_simple_2d_mesh():
    """Creates a simple 2D mesh with 9 quadrilateral cells for testing."""
    m = Mesh()
    m.dimension = 2

    # Create a 4x4 grid of nodes
    x = np.linspace(0, 1, 5)
    y = np.linspace(0, 1, 5)
    xx, yy = np.meshgrid(x, y)
    nodes = np.vstack([xx.ravel(), yy.ravel(), np.zeros(xx.size)]).T
    m.node_coords = nodes
    m.num_nodes = m.node_coords.shape[0]

    # Create 3x3 grid of quad cells
    cell_connectivity = []
    for j in range(4):
        for i in range(4):
            # Node indices for the current cell
            n0 = j * 5 + i
            n1 = j * 5 + i + 1
            n2 = (j + 1) * 5 + i + 1
            n3 = (j + 1) * 5 + i
            cell_connectivity.append([n0, n1, n2, n3])

    m.cell_connectivity = cell_connectivity
    m.num_cells = len(cell_connectivity)
    m.analyze_mesh()
    # m.print_summary()
    # m.plot()

    return m


def make_complex_2d_mesh():
    """Test the generation of a structured mesh."""
    projName = "rectangular_structured_mesh"
    gmsh.initialize()

    gmsh.model.add(projName)

    geom = Geometry(projName)
    surface_tag = geom.rectangle(length=100, width=100, mesh_size=5)

    mesher = MeshGenerator(surface_tags=surface_tag, output_dir="output_partition")
    mesh_filename = "structured_mesh.msh"
    mesh_params = {surface_tag: {"mesh_type": "tri", "char_length": 5}}
    mesher.generate(
        mesh_params=mesh_params,
        filename=mesh_filename,
        show_nodes=True,
        show_cells=True,
    )
    gmsh.finalize()

    mesh = Mesh()
    mesh.read_gmsh(str(os.path.join("output_partition", mesh_filename)))
    mesh.analyze_mesh()
    mesh.print_summary()

    return mesh


class TestPartitionMesh(unittest.TestCase):

    def setUp(self):
        self.tmp_path = "results/partition"
        os.makedirs(self.tmp_path, exist_ok=True)

    def tearDown(self):
        pass

    def test_metis_partitioning(self):
        """Test METIS partitioning."""
        mesh = make_complex_2d_mesh()
        n_parts = 4
        result = partition_mesh(mesh, n_parts, method="metis")
        result.print_summary()
        self.assertEqual(result.parts.shape[0], mesh.num_cells)
        self.assertEqual(len(np.unique(result.parts)), n_parts)
        mesh.plot(
            os.path.join(self.tmp_path, "mesh_partition_metis.png"),
            parts=result.parts,
        )

        halo_indices = result.halo_indices
        self.assertIsInstance(halo_indices, dict)
        self.assertEqual(len(halo_indices), n_parts)

        for rank in range(n_parts):
            self.assertIn(rank, halo_indices)
            self.assertIn("owned_cells", halo_indices[rank])
            self.assertIn("send", halo_indices[rank])
            self.assertIn("recv", halo_indices[rank])

    def test_hierarchical_partitioning(self):
        """Test hierarchical partitioning."""
        mesh = make_complex_2d_mesh()
        n_parts = 4
        result = partition_mesh(mesh, n_parts, method="hierarchical")
        result.print_summary()
        self.assertEqual(result.parts.shape[0], mesh.num_cells)
        self.assertEqual(len(np.unique(result.parts)), n_parts)
        mesh.plot(
            os.path.join(self.tmp_path, "mesh_partition_hierarchical.png"),
            parts=result.parts,
        )

        halo_indices = result.halo_indices
        self.assertIsInstance(halo_indices, dict)
        self.assertEqual(len(halo_indices), n_parts)

        for rank in range(n_parts):
            self.assertIn(rank, halo_indices)
            self.assertIn("owned_cells", halo_indices[rank])
            self.assertIn("send", halo_indices[rank])
            self.assertIn("recv", halo_indices[rank])


if __name__ == "__main__":
    unittest.main()
