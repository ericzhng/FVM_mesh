import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee

from .core_mesh import CoreMesh

# --- Helper for Adjacency Matrix ---


def _get_adjacency(mesh: CoreMesh) -> csr_matrix:
    """Builds the cell-to-cell adjacency matrix for the mesh."""
    if not hasattr(mesh, "cell_neighbors") or not mesh.cell_neighbors.size > 0:
        mesh.extract_neighbors()

    if mesh.num_cells == 0:
        return csr_matrix((0, 0), dtype=int)

    row, col = [], []
    for i in range(mesh.num_cells):
        for neighbor in mesh.cell_neighbors[i]:
            if neighbor != -1:
                row.append(i)
                col.append(neighbor)

    return csr_matrix(
        (np.ones_like(row), (row, col)), shape=(mesh.num_cells, mesh.num_cells)
    )


# --- Cell Reordering Strategies ---


class _CellReorderStrategy:
    """Abstract base class for cell reordering strategies."""

    def get_order(self, mesh: CoreMesh) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement this method.")


class _RCMStrategy(_CellReorderStrategy):
    """Reverse Cuthill-McKee ordering strategy."""

    def get_order(self, mesh: CoreMesh) -> np.ndarray:
        adj_matrix = _get_adjacency(mesh)
        return reverse_cuthill_mckee(adj_matrix)


class _GPSStrategy(_CellReorderStrategy):
    """Gibbs-Poole-Stockmeyer ordering strategy."""

    def get_order(self, mesh: CoreMesh) -> np.ndarray:
        from scipy.sparse.csgraph import breadth_first_order

        adj_matrix = _get_adjacency(mesh)
        if (
            adj_matrix is None
            or not hasattr(adj_matrix, "shape")
            or adj_matrix.shape is None
        ):
            return np.array([], dtype=int)
        n = adj_matrix.shape[0]
        if n == 0:
            return np.array([], dtype=int)

        degrees = adj_matrix.getnnz(axis=1)
        start_node = np.argmin(degrees)
        order, _ = breadth_first_order(
            adj_matrix, start_node, directed=False, return_predecessors=True
        )

        if not order.size:
            return reverse_cuthill_mckee(adj_matrix)  # Fallback

        x = order[-1]
        order, pred = breadth_first_order(
            adj_matrix, x, directed=False, return_predecessors=True
        )

        path = []
        if order.size > 0:
            curr = order[-1]
            while curr != -9999:
                path.append(curr)
                if curr < len(pred):
                    curr = pred[curr]
                else:
                    break

        center_node = path[len(path) // 2] if path else x

        q, visited, cm_order = [center_node], {center_node}, []
        while len(cm_order) < n:
            if not q:
                remaining = np.setdiff1d(np.arange(n), cm_order, assume_unique=True)
                if not remaining.size:
                    break
                min_deg_node = remaining[np.argmin(degrees[remaining])]
                q.append(min_deg_node)
                visited.add(min_deg_node)

            current = q.pop(0)
            cm_order.append(current)

            neighbors = sorted(
                [n for n in adj_matrix[current].indices if n not in visited],
                key=lambda n: degrees[n],
            )
            visited.update(neighbors)
            q.extend(neighbors)

        return np.array(cm_order, dtype=int)[::-1]


class _SloanStrategy(_CellReorderStrategy):
    """Sloan ordering strategy."""

    def get_order(self, mesh: CoreMesh) -> np.ndarray:
        from scipy.sparse.csgraph import dijkstra

        adj_matrix = _get_adjacency(mesh)
        if (
            adj_matrix is None
            or not hasattr(adj_matrix, "shape")
            or adj_matrix.shape is None
        ):
            return np.array([], dtype=int)
        n = adj_matrix.shape[0]
        if n == 0:
            return np.array([], dtype=int)

        degrees = adj_matrix.getnnz(axis=1)
        current_degrees = degrees.copy()

        start_node = np.argmin(degrees)
        end_node_distances = dijkstra(adj_matrix, indices=start_node, unweighted=True)
        end_node = np.argmax(end_node_distances)
        distances = dijkstra(adj_matrix, indices=end_node, unweighted=True)

        perm = np.full(n, -1, dtype=int)
        status = np.zeros(n, dtype=int)  # 0: inactive, 1: preactive, 2: numbered
        perm_counter = 0
        status[start_node] = 1

        while perm_counter < n:
            preactive_nodes = np.where(status == 1)[0]
            if not preactive_nodes.any():
                inactive_nodes = np.where(status == 0)[0]
                if not inactive_nodes.any():
                    break
                new_start = inactive_nodes[np.argmin(current_degrees[inactive_nodes])]
                status[new_start] = 1
                preactive_nodes = np.array([new_start])

            priorities = (
                1 * distances[preactive_nodes] - 2 * current_degrees[preactive_nodes]
            )
            next_node = preactive_nodes[np.argmax(priorities)]

            perm[perm_counter] = next_node
            status[next_node] = 2
            perm_counter += 1

            for neighbor in adj_matrix[next_node].indices:
                if status[neighbor] == 0:
                    status[neighbor] = 1
                if status[neighbor] != 2:
                    current_degrees[neighbor] -= 1
        return perm


class _SpectralStrategy(_CellReorderStrategy):
    """Spectral bisection ordering strategy."""

    def get_order(self, mesh: CoreMesh) -> np.ndarray:
        from scipy.sparse.linalg import eigsh
        from scipy.sparse import csgraph

        adj_matrix = _get_adjacency(mesh)
        laplacian = csgraph.laplacian(adj_matrix, normed=False)
        _, eigenvectors = eigsh(laplacian, k=2, which="SM")
        return np.argsort(eigenvectors[:, 1])


class _SpatialXStrategy(_CellReorderStrategy):
    """Sorts cells by their centroid's x-coordinate."""

    def get_order(self, mesh: CoreMesh) -> np.ndarray:
        if not hasattr(mesh, "cell_centroids") or not mesh.cell_centroids.size > 0:
            mesh._compute_centroids()
        return np.argsort(mesh.cell_centroids[:, 0])


class _SpatialYStrategy(_CellReorderStrategy):
    """Sorts cells by their centroid's y-coordinate."""

    def get_order(self, mesh: CoreMesh) -> np.ndarray:
        if not hasattr(mesh, "cell_centroids") or not mesh.cell_centroids.size > 0:
            mesh._compute_centroids()
        return np.argsort(mesh.cell_centroids[:, 1])


class _RandomStrategy(_CellReorderStrategy):
    """Randomly permutes the cells."""

    def get_order(self, mesh: CoreMesh) -> np.ndarray:
        return np.random.permutation(mesh.num_cells)


def renumber_cells(mesh: CoreMesh, strategy: str = "rcm") -> None:
    """
    Renumbers the cells of a mesh in-place to optimize matrix bandwidth.

    This function modifies the mesh object directly.
    If the mesh is a LocalMesh, only the owned cells are reordered.
    Derived properties will need to be re-computed.

    Args:
        mesh: The input CoreMesh object to be modified.
        strategy: The renumbering strategy. Options include 'rcm', 'gps',
                  'sloan', 'spectral', 'spatial_x', 'spatial_y', 'random'.
    """
    from .local_mesh import LocalMesh

    is_local = isinstance(mesh, LocalMesh)

    num_to_reorder = mesh.num_owned_cells if is_local else mesh.num_cells

    if num_to_reorder == 0:
        return

    strategy_map = {
        "rcm": _RCMStrategy,
        "gps": _GPSStrategy,
        "sloan": _SloanStrategy,
        "spectral": _SpectralStrategy,
        "spatial_x": _SpatialXStrategy,
        "spatial_y": _SpatialYStrategy,
        "random": _RandomStrategy,
    }

    if strategy not in strategy_map:
        raise NotImplementedError(
            f"Cell renumbering strategy '{strategy}' is not implemented."
        )

    reorder_strategy = strategy_map[strategy]()

    if is_local:
        # Create a temporary mesh view for the owned cells
        owned_mesh = CoreMesh()
        owned_mesh.num_cells = mesh.num_owned_cells
        owned_mesh.cell_connectivity = mesh.cell_connectivity[: mesh.num_owned_cells]
        owned_mesh.node_coords = mesh.node_coords  # Needed for some strategies
        owned_mesh.dimension = mesh.dimension

        # We need to compute neighbors only for the owned part
        owned_mesh._extract_neighbors()
        if hasattr(mesh, "cell_centroids") and mesh.cell_centroids.size > 0:
            owned_mesh.cell_centroids = mesh.cell_centroids[: mesh.num_owned_cells]

        new_order_local = reorder_strategy.get_order(owned_mesh)

        # The new order is for the owned cells. We need to combine it with the halo cells.
        new_order = np.concatenate(
            (new_order_local, np.arange(mesh.num_owned_cells, mesh.num_cells))
        )

    else:
        new_order = reorder_strategy.get_order(mesh)

    # Reorder cell properties in-place
    # It's safer to create a new list and then assign it back
    new_cell_connectivity = [mesh.cell_connectivity[i] for i in new_order]
    mesh.cell_connectivity = new_cell_connectivity

    if hasattr(mesh, "cell_type_ids") and mesh.cell_type_ids.size > 0:
        mesh.cell_type_ids = mesh.cell_type_ids[new_order]

    # Invalidate derived fields that depend on cell ordering
    if hasattr(mesh, "cell_neighbors"):
        mesh.cell_neighbors = np.array([])
    if hasattr(mesh, "cell_centroids"):
        # we need to reorder centroids as well if they exist
        if mesh.cell_centroids.size > 0:
            mesh.cell_centroids = mesh.cell_centroids[new_order]
    if hasattr(mesh, "_is_analyzed"):
        setattr(mesh, "_is_analyzed", False)


# --- Node Reordering ---


def renumber_nodes(mesh: CoreMesh, strategy: str = "rcm") -> None:
    """
    Renumbers the nodes of a mesh in-place to optimize matrix bandwidth.

    This function modifies the mesh object directly.

    Args:
        mesh: The input CoreMesh object to be modified.
        strategy: The renumbering strategy. Options include 'rcm', 'sequential',
                  'reverse', 'spatial_x', 'spatial_y', 'random'.
    """
    if mesh.num_nodes == 0:
        return

    # For node reordering, the adjacency is node-to-node
    rows, cols = [], []
    for cell in mesh.cell_connectivity:
        for i in range(len(cell)):
            for j in range(i + 1, len(cell)):
                rows.extend([cell[i], cell[j]])
                cols.extend([cell[j], cell[i]])
    adj_matrix = csr_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(mesh.num_nodes, mesh.num_nodes)
    )

    if strategy == "rcm":
        new_order = reverse_cuthill_mckee(adj_matrix)
    elif strategy == "sequential":
        new_order = np.arange(mesh.num_nodes, dtype=int)
    elif strategy == "reverse":
        new_order = np.arange(mesh.num_nodes - 1, -1, -1, dtype=int)
    elif strategy == "random":
        new_order = np.random.permutation(mesh.num_nodes)
    elif strategy == "spatial_x":
        new_order = np.argsort(mesh.node_coords[:, 0])
    elif strategy == "spatial_y":
        new_order = np.argsort(mesh.node_coords[:, 1])
    else:
        raise NotImplementedError(
            f"Node renumbering strategy '{strategy}' is not implemented."
        )

    # Create the inverse mapping (old index -> new index)
    remap = np.empty_like(new_order)
    remap[new_order] = np.arange(mesh.num_nodes, dtype=int)

    # Apply renumbering in-place
    mesh.node_coords = mesh.node_coords[new_order]
    mesh.cell_connectivity = [
        list(remap[np.array(c, dtype=int)]) for c in mesh.cell_connectivity
    ]

    # Invalidate derived fields that depend on node/cell ordering
    if hasattr(mesh, "cell_neighbors"):
        mesh.cell_neighbors = np.array([])
    if hasattr(mesh, "cell_centroids"):
        mesh.cell_centroids = np.array([])
    if hasattr(mesh, "_is_analyzed"):
        setattr(mesh, "_is_analyzed", False)
