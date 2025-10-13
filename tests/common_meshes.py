import os

import numpy as np
from polymesh.poly_mesh import PolyMesh


def create_structured_quad_mesh(nx: int, ny: int):
    """
    Creates a structured quadrilateral mesh of size nx x ny.

    Args:
        nx (int): Number of cells in the x-direction.
        ny (int): Number of cells in the y-direction.

    Returns:
        PolyMesh: The generated mesh.
    """
    mesh = PolyMesh()
    mesh.dimension = 2

    num_nodes_x = nx + 1
    num_nodes_y = ny + 1
    mesh.num_nodes = num_nodes_x * num_nodes_y
    mesh.num_cells = nx * ny

    # Generate node coordinates
    node_coords = []
    for j in range(num_nodes_y):
        for i in range(num_nodes_x):
            node_coords.append([float(i), float(j), 0.0])
    mesh.node_coords = np.array(node_coords)

    # Generate cell connectivity
    cell_connectivity = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * num_nodes_x + i
            n1 = j * num_nodes_x + (i + 1)
            n2 = (j + 1) * num_nodes_x + (i + 1)
            n3 = (j + 1) * num_nodes_x + i
            cell_connectivity.append([n0, n1, n2, n3])
    mesh.cell_connectivity = cell_connectivity
    mesh.analyze_mesh()

    return mesh


def create_2x2_quad_mesh_fixture():
    """
    Provides a 2x2 quadrilateral mesh fixture for testing.

    Returns:
        tuple: A tuple containing:
            - PolyMesh: The 2x2 mesh.
            - np.ndarray: The partitioning array.
            - int: The number of partitions.
    """
    mesh = create_structured_quad_mesh(2, 2)

    # Partitioning (2 parts):
    # - Rank 0 owns cells [0, 1]
    # - Rank 1 owns cells [2, 3]
    parts = np.array([0, 0, 1, 1])
    n_parts = 2

    return mesh, parts, n_parts


def create_3x3_quad_mesh_fixture():
    """
    Provides a 3x3 quadrilateral mesh fixture for testing.

    Returns:
        tuple: A tuple containing:
            - PolyMesh: The 3x3 mesh.
            - np.ndarray: The partitioning array.
            - int: The number of partitions.
    """
    mesh = create_structured_quad_mesh(3, 3)

    # Partitioning (3 parts):
    # - Rank 0 owns cells [0, 1, 2]
    # - Rank 1 owns cells [3, 4, 5]
    # - Rank 2 owns cells [6, 7, 8]
    parts = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    n_parts = 3

    return mesh, parts, n_parts


def make_test_mesh():
    """Creates a test mesh from a file."""
    msh_file = os.path.join(
        os.path.dirname(__file__), "..", "data", "sample_mixed_mesh.msh"
    )
    mesh = PolyMesh()
    mesh.read_gmsh(msh_file)
    mesh.analyze_mesh()
    return mesh
