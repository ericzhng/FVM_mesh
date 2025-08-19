import os
import json
from typing import List, Tuple, Optional

os.environ["METIS_DLL"] = os.path.join(os.getcwd(), "dll", "metis.dll")

import numpy as np

try:
    import metis  # type: ignore
except ImportError:
    metis = None

from src.mesh_analysis import Mesh2D


def partition_mesh(
    mesh: Mesh2D,
    n_parts: int,
    method: str = "metis",
    cell_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Partition mesh elements into n_parts.

    Args:
        mesh: The mesh object to partition.
        n_parts: The number of partitions.
        method: The partitioning method ('metis', 'hierarchical').
        cell_weights: Optional weights for each cell.

    Returns:
        An array of partition indices for each cell.
    """
    if n_parts <= 1:
        return np.zeros(mesh.num_cells, dtype=int)

    if not mesh._is_analyzed:
        mesh.analyze_mesh()

    adjacency = _get_adjacency(mesh)

    if method == "metis":
        return _partition_with_metis(adjacency, n_parts, cell_weights)
    elif method == "hierarchical":
        return _partition_with_hierarchical(mesh, n_parts, cell_weights)
    else:
        raise NotImplementedError(f"Partition method '{method}' not implemented")


def _get_adjacency(mesh: Mesh2D) -> List[List[int]]:
    """Computes the adjacency list for the mesh cells."""
    adjacency: List[List[int]] = []
    for i in range(mesh.num_cells):
        neighs = {int(nb) for nb in mesh.cell_neighbors[i] if nb != -1}
        neighs.discard(i)
        adjacency.append(sorted(neighs))
    return adjacency


def _partition_with_metis(
    adjacency: List[List[int]], n_parts: int, cell_weights: Optional[np.ndarray]
) -> np.ndarray:
    """Partitions the mesh using METIS."""
    if metis is None:
        raise ImportError("metis python binding not available")
    try:
        if cell_weights is not None:
            try:
                _, parts = metis.part_graph(
                    adjacency, nparts=n_parts, vwgt=cell_weights.tolist()
                )
            except TypeError:
                _, parts = metis.part_graph(
                    adjacency, nparts=n_parts, vweights=cell_weights.tolist()
                )
        else:
            _, parts = metis.part_graph(adjacency, nparts=n_parts)
        return np.array(parts, dtype=int)
    except Exception as ex:
        raise RuntimeError(f"METIS partitioning failed: {ex}")


def _partition_with_hierarchical(
    mesh: Mesh, n_parts: int, cell_weights: Optional[np.ndarray]
) -> np.ndarray:
    """Partitions the mesh using a hierarchical coordinate bisection method."""
    centroids = mesh.cell_centroids
    weights = (
        cell_weights
        if cell_weights is not None
        else np.ones(mesh.num_cells, dtype=float)
    )
    parts = -np.ones(mesh.num_cells, dtype=int)

    def recurse(idxs: np.ndarray, part_ids: List[int]):
        if len(part_ids) == 1:
            parts[idxs] = part_ids[0]
            return
        pts = centroids[idxs]
        spans = pts.max(axis=0) - pts.min(axis=0)
        axis = int(np.argmax(spans))
        order = np.argsort(pts[:, axis])
        w = weights[idxs][order]
        cum = np.cumsum(w)
        total = cum[-1]
        split = int(np.searchsorted(cum, total / 2.0))
        if split == 0 or split == len(order):
            split = len(order) // 2
        left = idxs[order[:split]]
        right = idxs[order[split:]]
        mid = len(part_ids) // 2
        recurse(left, part_ids[:mid])
        recurse(right, part_ids[mid:])

    all_idx = np.arange(mesh.num_cells, dtype=int)
    recurse(all_idx, list(range(n_parts)))
    return parts


def reconstruct_mesh_from_decomposed_dir(decomposed_dir: str) -> Mesh2D:
    """Reconstruct a global mesh from processor*/local_mesh.npz outputs."""
    proc_dirs = sorted(
        [d for d in os.listdir(decomposed_dir) if d.startswith("processor")]
    )
    node_blocks: List[Tuple[np.ndarray, np.ndarray]] = []
    global_cells: List[List[int]] = []
    for proc in proc_dirs:
        path = os.path.join(decomposed_dir, proc, "node_coords.npy")
        if not os.path.exists(path):
            continue
        gnode_ids = np.load(os.path.join(decomposed_dir, proc, "global_node_ids.npy"))
        local_coords = np.load(os.path.join(decomposed_dir, proc, "node_coords.npy"))
        local_conn = np.load(
            os.path.join(decomposed_dir, proc, "cell_connectivity.npy"),
            allow_pickle=True,
        )
        node_blocks.append((gnode_ids, local_coords))

        for conn in local_conn:
            global_conn = [int(gnode_ids[int(li)]) for li in conn]
            global_cells.append(global_conn)

    if not node_blocks:
        raise RuntimeError("No processor data found")

    all_global = np.concatenate([b[0] for b in node_blocks])
    unique_global, _ = np.unique(all_global, return_inverse=True)
    id_to_idx = {int(g): idx for idx, g in enumerate(unique_global)}
    coords_dim = node_blocks[0][1].shape[1]
    global_coords = np.zeros((len(unique_global), coords_dim))

    for g_ids, coords in node_blocks:
        for i, gid in enumerate(g_ids):
            idx = id_to_idx[int(gid)]
            global_coords[idx, :] = coords[i]

    new_mesh = Mesh()
    new_mesh.node_coords = global_coords
    new_mesh.num_nodes = global_coords.shape[0]
    new_mesh.cell_connectivity = [
        [id_to_idx[int(g)] for g in gc] for gc in global_cells
    ]
    new_mesh.num_cells = len(new_mesh.cell_connectivity)
    new_mesh.analyze_mesh()
    return new_mesh


def write_decomposed_mesh(
    mesh: Mesh2D, elem_parts: np.ndarray, output_dir: str, n_parts: int
) -> None:
    """Write per-processor directories with numpy and json outputs."""
    if elem_parts is None or len(elem_parts) != mesh.num_cells:
        raise RuntimeError("Element partitioning not computed or invalid.")
    os.makedirs(output_dir, exist_ok=True)
    for p in range(n_parts):
        proc_dir = os.path.join(output_dir, f"processor{p}")
        os.makedirs(proc_dir, exist_ok=True)

        mask = elem_parts == p
        local_cell_indices = np.nonzero(mask)[0]

        if local_cell_indices.size == 0:
            coords_shape = (0, mesh.node_coords.shape[1] if mesh.num_nodes > 0 else 3)
            local_coords = np.empty(coords_shape)
            local_conn = np.empty((0,), dtype=object)
            unique_nodes = np.array([], dtype=int)
        else:
            local_cells = [mesh.cell_connectivity[i] for i in local_cell_indices]
            all_nodes = np.hstack([np.array(c, dtype=int) for c in local_cells])
            unique_nodes = np.unique(all_nodes)
            local_node_map = {int(g): i for i, g in enumerate(unique_nodes)}
            local_coords = mesh.node_coords[unique_nodes]
            local_conn = [[local_node_map[int(g)] for g in c] for c in local_cells]

        _write_partition_files(
            proc_dir,
            p,
            mesh,
            local_cell_indices,
            local_coords,
            local_conn,
            unique_nodes,
        )


def _write_partition_files(
    proc_dir: str,
    part_id: int,
    mesh: Mesh2D,
    local_cell_indices: np.ndarray,
    local_coords: np.ndarray,
    local_conn: List[List[int]],
    unique_nodes: np.ndarray,
):
    """Helper to write files for a single partition."""
    local_boundary_faces, local_boundary_tags = _extract_local_boundary(
        mesh, local_cell_indices, unique_nodes
    )

    np.save(os.path.join(proc_dir, "node_coords.npy"), local_coords)
    np.save(os.path.join(proc_dir, "global_node_ids.npy"), unique_nodes)
    np.save(
        os.path.join(proc_dir, "cell_connectivity.npy"),
        np.array(local_conn, dtype=object),
    )
    np.save(os.path.join(proc_dir, "global_cell_indices.npy"), local_cell_indices)
    np.save(os.path.join(proc_dir, "boundary_faces_nodes.npy"), local_boundary_faces)
    np.save(os.path.join(proc_dir, "boundary_faces_tags.npy"), local_boundary_tags)

    meta = {
        "elem_part": int(part_id),
        "num_local_nodes": int(local_coords.shape[0]),
        "num_local_cells": int(local_cell_indices.size),
    }
    with open(os.path.join(proc_dir, "mesh.json"), "w") as fh:
        json.dump(meta, fh, indent=2)


def _extract_local_boundary(
    mesh: Mesh2D, local_cell_indices: np.ndarray, unique_nodes: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts boundary faces and tags for a local partition."""
    if (
        getattr(mesh, "boundary_faces_nodes", None) is None
        or mesh.boundary_faces_nodes.size == 0
    ):
        return np.empty((0, 0), dtype=int), np.empty((0,), dtype=int)

    if local_cell_indices.size == 0:
        is_in = np.zeros(mesh.boundary_faces_nodes.shape[0], dtype=bool)
    else:
        local_node_set = set(unique_nodes.tolist())
        is_in = np.array(
            [
                all(int(n) in local_node_set for n in face)
                for face in mesh.boundary_faces_nodes
            ],
            dtype=bool,
        )

    sel_idx = np.nonzero(is_in)[0]
    if sel_idx.size:
        local_node_map = {int(g): i for i, g in enumerate(unique_nodes)}
        local_boundary_faces = np.array(
            [
                [local_node_map[int(n)] for n in mesh.boundary_faces_nodes[i]]
                for i in sel_idx
            ],
            dtype=int,
        )
        local_boundary_tags = mesh.boundary_faces_tags[sel_idx]
    else:
        shape = (
            (0, mesh.boundary_faces_nodes.shape[1])
            if mesh.boundary_faces_nodes.size
            else (0, 0)
        )
        local_boundary_faces = np.empty(shape, dtype=int)
        local_boundary_tags = np.empty((0,), dtype=int)

    return local_boundary_faces, local_boundary_tags


def write_decomposed_mesh_gmsh(
    output_dir: str, n_parts: int, msh_filename: str = "processor{p}.msh"
) -> None:
    """Writes a .msh file for each partition's decomposed mesh data."""
    for p in range(n_parts):
        proc_dir = os.path.join(output_dir, f"processor{p}")
        if not os.path.isdir(proc_dir):
            continue

        node_coords_path = os.path.join(proc_dir, "node_coords.npy")
        cell_conn_path = os.path.join(proc_dir, "cell_connectivity.npy")

        if not os.path.exists(node_coords_path) or not os.path.exists(cell_conn_path):
            continue

        node_coords = np.load(node_coords_path)
        cell_conn = np.load(cell_conn_path, allow_pickle=True)

        out_path = os.path.join(proc_dir, msh_filename.format(p=p))
        with open(out_path, "w") as fh:
            fh.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
            fh.write(f"$Nodes\n{node_coords.shape[0]}\n")
            for i, xyz in enumerate(node_coords, start=1):
                fh.write(f"{i} {xyz[0]} {xyz[1]} {xyz[2]}\n")
            fh.write("$EndNodes\n")
            fh.write(f"$Elements\n{len(cell_conn)}\n")
            eid = 1
            for conn in cell_conn:
                nn = len(conn)
                etype = 4 if nn == 4 else (2 if nn == 3 else 4)  # Basic type mapping
                nodes_str = " ".join(str(int(ci) + 1) for ci in conn)
                fh.write(f"{eid} {etype} 0 {nodes_str}\n")
                eid += 1
            fh.write("$EndElements\n")
