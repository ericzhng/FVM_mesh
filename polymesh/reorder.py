import copy

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee

from polymesh.core_mesh import CoreMesh


def renumber_nodes(mesh: CoreMesh, strategy: str = "sequential") -> CoreMesh:
    """
    Renumbers the nodes of a mesh globally based on a specified strategy.

    This function creates a new mesh with renumbered nodes, leaving the original
    mesh unmodified.

    Args:
        mesh (Mesh): The input mesh object.
        strategy (str): The renumbering strategy. Can be one of:
                        'sequential', 'reverse', 'spatial_x', 'random'.

    Returns:
        Mesh: A new mesh object with renumbered nodes.
    """
    if mesh.num_nodes == 0:
        return copy.deepcopy(mesh)

    if strategy == "sequential":
        new_order = np.arange(mesh.num_nodes, dtype=int)
    elif strategy == "reverse":
        new_order = np.arange(mesh.num_nodes - 1, -1, -1, dtype=int)
    elif strategy == "random":
        new_order = np.random.permutation(mesh.num_nodes)
    elif strategy == "spatial_x":
        new_order = np.argsort(mesh.node_coords[:, 0])
    else:
        raise NotImplementedError(
            f"renumbering strategy '{strategy}' is not implemented"
        )

    # Create a deep copy to avoid modifying the original mesh
    new_mesh = copy.deepcopy(mesh)

    # Create the inverse mapping (old index -> new index)
    remap = np.empty_like(new_order)
    remap[new_order] = np.arange(mesh.num_nodes, dtype=int)

    # Apply the renumbering to the new mesh object
    new_mesh.node_coords = new_mesh.node_coords[new_order]
    new_mesh.cell_connectivity = [
        list(remap[np.array(c, dtype=int)]) for c in new_mesh.cell_connectivity
    ]

    if (
        hasattr(new_mesh, "boundary_faces_nodes")
        and new_mesh.boundary_faces_nodes.size > 0
    ):
        new_mesh.boundary_faces_nodes = remap[new_mesh.boundary_faces_nodes]

    # Since node numbering has changed, derived geometric fields are invalid.
    # Clearing them ensures that they are re-computed when needed.
    new_mesh.cell_faces = []
    new_mesh.cell_neighbors = np.array([])
    new_mesh.cell_centroids = np.array([])
    new_mesh.face_midpoints = np.array([])
    new_mesh.face_normals = np.array([])
    new_mesh.face_areas = np.array([])
    new_mesh.cell_volumes = np.array([])
    new_mesh._is_analyzed = False  # Mark as not analyzed

    return new_mesh


def _get_adjacency(mesh: CoreMesh) -> csr_matrix:
    """Builds the cell adjacency matrix."""
    if mesh.num_cells == 0:
        return csr_matrix((0, 0), dtype=int)
    row = []
    col = []
    for i in range(mesh.num_cells):
        for neighbor in mesh.cell_neighbors[i]:
            if neighbor != -1:
                row.append(i)
                col.append(neighbor)
    return csr_matrix(
        (np.ones_like(row), (row, col)), shape=(mesh.num_cells, mesh.num_cells)
    )


def _spectral_ordering(adj_matrix: csr_matrix) -> np.ndarray:
    """Computes cell ordering using the Fiedler vector (spectral bisection)."""
    if adj_matrix is None:
        raise ValueError("Input adjacency matrix cannot be None.")
    from scipy.sparse.linalg import eigsh
    from scipy.sparse import csgraph

    laplacian = csgraph.laplacian(adj_matrix, normed=False)
    # Find the eigenvector corresponding to the second smallest eigenvalue
    # k=2 to get the first two, v0 is for the smallest (0), v1 for the second smallest
    _, eigenvectors = eigsh(laplacian, k=2, which="SM", tol=2)
    fiedler_vector = eigenvectors[:, 1]
    return np.argsort(fiedler_vector)


def _sloan_ordering(adj_matrix: csr_matrix) -> np.ndarray:
    """Computes cell ordering using the Sloan algorithm."""

    if adj_matrix is None:
        raise ValueError("Input adjacency matrix cannot be None.")

    from scipy.sparse.csgraph import dijkstra

    n = adj_matrix.shape[0]
    if n == 0:
        return np.array([], dtype=int)

    degrees = adj_matrix.getnnz(axis=1)
    current_degrees = degrees.copy()

    # 1. Find start and end nodes
    start_node = np.argmin(degrees)
    end_node_distances = dijkstra(adj_matrix, indices=start_node, unweighted=True)
    end_node = np.argmax(end_node_distances)

    # Calculate distances from the end node for the priority function
    distances = dijkstra(adj_matrix, indices=end_node, unweighted=True)

    # 2. Initialization
    perm = np.full(n, -1, dtype=int)
    status = np.zeros(n, dtype=int)  # 0: inactive, 1: preactive, 2: numbered

    W1 = 1  # Weight for distance
    W2 = 2  # Weight for degree

    # 3. Start algorithm
    perm_counter = 0
    status[start_node] = 1  # Start node is preactive

    while perm_counter < n:
        preactive_nodes = np.where(status == 1)[0]
        if not preactive_nodes.any():
            # Handle disconnected graphs
            inactive_nodes = np.where(status == 0)[0]
            if not inactive_nodes.any():
                break  # All done
            # Find new start node from remaining ones
            new_start_node = inactive_nodes[np.argmin(current_degrees[inactive_nodes])]
            status[new_start_node] = 1
            preactive_nodes = np.array([new_start_node])

        # Find node with highest priority to number next
        priorities = (
            W1 * distances[preactive_nodes] - W2 * current_degrees[preactive_nodes]
        )
        next_node_local_idx = np.argmax(priorities)
        next_node = preactive_nodes[next_node_local_idx]

        # Number the selected node
        perm[perm_counter] = next_node
        status[next_node] = 2  # numbered
        perm_counter += 1

        # Update neighbors
        neighbors = adj_matrix[next_node].indices
        for neighbor in neighbors:
            if status[neighbor] == 0:  # if inactive
                status[neighbor] = 1  # becomes preactive
            if status[neighbor] != 2:  # if not numbered
                current_degrees[neighbor] -= 1

    return perm


def _gps_ordering(adj_matrix: csr_matrix) -> np.ndarray:
    """Computes cell ordering using the Gibbs-Poole-Stockmeyer algorithm."""

    if adj_matrix is None:
        raise ValueError("Input adjacency matrix cannot be None.")

    from scipy.sparse.csgraph import breadth_first_order, reverse_cuthill_mckee

    n = adj_matrix.shape[0]
    if n == 0:
        return np.array([], dtype=int)

    degrees = adj_matrix.getnnz(axis=1)

    # Find a node with minimum degree
    start_node = np.argmin(degrees)

    # Find pseudo-peripheral node
    order = breadth_first_order(
        adj_matrix, start_node, directed=False, return_predecessors=False
    )

    if not order.size:  # Handle disconnected graph
        # Fallback to default RCM if BFS fails
        return reverse_cuthill_mckee(adj_matrix)

    x = order[-1]  # Last node in BFS is a candidate

    order, pred = breadth_first_order(
        adj_matrix, x, directed=False, return_predecessors=True
    )

    # Find path back to start
    path = []
    if order.size > 0:
        curr = order[-1]  # End of the diameter
        # Traverse predecessors until we find the root (-9999)
        while curr != -9999:
            path.append(curr)
            # Check if curr is in the pred array's indices
            if curr < len(pred):
                curr = pred[curr]
            else:
                # This can happen in disconnected graphs. Break the loop.
                break

    # Center of this path is a good starting point for CM
    if path:
        center_node = path[len(path) // 2]
    else:
        center_node = x  # Fallback to the candidate

    # Manual Cuthill-McKee implementation since start_node is not supported
    q = [center_node]
    visited = {center_node}
    cm_order = []

    while len(cm_order) < n:
        if not q:
            # Handle disconnected graphs
            remaining_mask = np.ones(n, dtype=bool)
            remaining_mask[cm_order] = False
            remaining_indices = np.where(remaining_mask)[0]
            if not remaining_indices.size:
                break
            # Find a new starting node with minimum degree among the remaining
            min_deg_node = remaining_indices[np.argmin(degrees[remaining_indices])]
            q.append(min_deg_node)
            visited.add(min_deg_node)

        current = q.pop(0)
        cm_order.append(current)

        neighbors = adj_matrix[current].indices
        unvisited_neighbors = [n for n in neighbors if n not in visited]

        # Sort neighbors by degree
        unvisited_neighbors.sort(key=lambda n: degrees[n])

        for neighbor in unvisited_neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                q.append(neighbor)

    # Reverse the order for RCM
    return np.array(cm_order, dtype=int)[::-1]


def renumber_cells(mesh: CoreMesh, strategy: str = "rcm") -> CoreMesh:
    """
    Renumbers the cells of a mesh to optimize matrix bandwidth for FVM solvers.

    This function creates a new mesh with renumbered cells, leaving the original
    mesh unmodified. It works by reordering the core cell connectivity and then
    re-running the mesh analysis to derive all other geometric properties.

    Args:
        mesh (Mesh): The input mesh object.
        strategy (str): The renumbering strategy. Can be one of:
                        'rcm' (Reverse Cuthill-McKee),
                        'gps' (Gibbs-Poole-Stockmeyer),
                        'sloan' (Sloan Algorithm),
                        'spectral' (Spectral Bisection),
                        'spatial_x' (Sort by x-coordinate),
                        'random' (Random permutation).

    Returns:
        Mesh: A new mesh object with renumbered cells.
    """
    if mesh.num_cells == 0:
        return copy.deepcopy(mesh)

    if not mesh._is_analyzed:
        mesh.analyze_mesh()

    # 1. Determine the new cell ordering based on the chosen strategy
    if strategy in ["rcm", "gps", "sloan", "spectral"]:
        adj_matrix = _get_adjacency(mesh)
        if strategy == "rcm":
            new_order = reverse_cuthill_mckee(adj_matrix)
        elif strategy == "gps":
            new_order = _gps_ordering(adj_matrix)
        elif strategy == "sloan":
            new_order = _sloan_ordering(adj_matrix)
        elif strategy == "spectral":
            new_order = _spectral_ordering(adj_matrix)

    elif strategy == "spatial_x":
        if mesh.cell_centroids.size == 0:
            mesh._compute_centroids()
        new_order = np.argsort(mesh.cell_centroids[:, 0])

    elif strategy == "random":
        new_order = np.random.permutation(mesh.num_cells)

    else:
        raise NotImplementedError(
            f"Cell renumbering strategy '{strategy}' is not implemented"
        )

    # 2. Create a new mesh with the reordered cell connectivity
    new_mesh = CoreMesh()
    new_mesh.dimension = mesh.dimension
    new_mesh.node_coords = mesh.node_coords.copy()
    new_mesh.num_nodes = mesh.num_nodes

    # Reorder core cell properties
    new_mesh.cell_connectivity = [mesh.cell_connectivity[i] for i in new_order]
    if mesh.cell_type_ids.size > 0:
        new_mesh.cell_type_ids = mesh.cell_type_ids[new_order]

    new_mesh.num_cells = mesh.num_cells
    new_mesh.cell_type_map = mesh.cell_type_map

    # 3. Re-analyze the new mesh to compute all derived properties
    new_mesh.analyze_mesh()

    return new_mesh
