"""
mesh.py - mesh handling and partition utilities for parallel FVM solver.

Features:
- Mesh class: holds geometry/connectivity; independent of partitioning.
- PartitionManager: partition elements (metis/scotch/hierarchical), renumbering, write per-processor outputs.
- Portable per-processor writer: JSON metadata + .npy binary arrays.
- Gmsh .msh writer (simple) for visualization of per-processor meshes.
- Halo builder: create local-to-global maps and neighbor send lists.
Compatible with Python 3.8+. External packages (gmsh, metis, scotch) are optional.

Key classes:
- Mesh: read + analyze geometry/connectivity. Independent of partitioning/renumbering.
- PartitionManager: partition elements (using METIS), perform renumbering, and write per-processor folders
  following an OpenFOAM-like decomposePar pattern (creates processorN/ with a .npz mesh dump).

Usage example (at bottom): read mesh -> analyze_mesh() -> PartitionManager.decompose_par(...)
"""

import os
import json
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

# Optional imports
try:
    import gmsh
except Exception:
    gmsh = None  # gmsh not available

# METIS binding optional
try:
    import metis  # type: ignore
except Exception:
    metis = None

# Try scotch bindings (optional)
_scotch_module = None
for _nm in ("scotch", "pyscotch"):
    try:
        _scotch_module = __import__(_nm)
        break
    except Exception:
        _scotch_module = None


class PartitionManager:
    """Manage partitioning, renumbering, and writing per-processor outputs.

    Responsibilities:
    - compute element partitioning using METIS
    - optional node renumbering per partition
    - write per-processor mesh dumps into `processor<p>` directories
    """

    def __init__(self, mesh: Mesh):
        self.mesh = mesh
        self.elem_parts: Optional[np.ndarray] = None

    def partition_elements(
        self,
        n_parts: int,
        method: str = "metis",
        cell_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Partition elements into n_parts. Returns array elem_parts of length num_cells."""
        if n_parts <= 1:
            self.elem_parts = np.zeros(self.mesh.num_cells, dtype=int)
            return self.elem_parts

        # ensure neighbors exist
        if self.mesh.cell_neighbors.size == 0:
            self.mesh._compute_centroids()
            self.mesh._extract_cell_faces()
            self.mesh._compute_cell_neighbors()

        adjacency: List[List[int]] = []
        for i in range(self.mesh.num_cells):
            neighs = {int(nb) for nb in self.mesh.cell_neighbors[i] if nb != -1}
            neighs.discard(i)
            adjacency.append(sorted(neighs))

        if method == "metis":
            if metis is None:
                raise ImportError("metis python binding not available")
            try:
                if cell_weights is not None:
                    # try common arg names
                    try:
                        _, parts = metis.part_graph(
                            adjacency, nparts=n_parts, vwgt=cell_weights.tolist()
                        )
                    except TypeError:
                        try:
                            _, parts = metis.part_graph(
                                adjacency,
                                nparts=n_parts,
                                vweights=cell_weights.tolist(),
                            )
                        except TypeError:
                            _, parts = metis.part_graph(adjacency, nparts=n_parts)
                else:
                    _, parts = metis.part_graph(adjacency, nparts=n_parts)
            except Exception as ex:
                raise RuntimeError(f"METIS partitioning failed: {ex}")
            self.elem_parts = np.array(parts, dtype=int)
            return self.elem_parts

        if method == "scotch":
            if _scotch_module is None:
                raise ImportError("scotch binding not available")
            # Best-effort attempt; actual bindings differ. Use hierarchical as fallback
            try:
                graph = _scotch_module.Graph()
                for i, nbrs in enumerate(adjacency):
                    graph.addVertex(i, nbrs)
                parts = graph.partition(n_parts)
                self.elem_parts = np.array(parts, dtype=int)
                return self.elem_parts
            except Exception:
                method = "hierarchical"  # fallback

        if method == "hierarchical":
            centroids = self.mesh.cell_centroids
            weights = (
                cell_weights
                if cell_weights is not None
                else np.ones(self.mesh.num_cells, dtype=float)
            )
            parts = -np.ones(self.mesh.num_cells, dtype=int)

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

            all_idx = np.arange(self.mesh.num_cells, dtype=int)
            recurse(all_idx, list(range(n_parts)))
            self.elem_parts = parts
            return self.elem_parts

        raise NotImplementedError(f"Partition method '{method}' not implemented")

    def renumber_nodes_global(self, strategy: str = "sequential") -> None:
        """A simple global node renumbering that reorders nodes in mesh.node_coords.

        This is separate from partitioning and may be called independently.
        Strategies: 'sequential', 'reverse', 'spatial_x', 'random'.
        """
        if self.mesh.num_nodes == 0:
            return
        if strategy == "sequential":
            new_order = np.arange(self.mesh.num_nodes, dtype=int)
        elif strategy == "reverse":
            new_order = np.arange(self.mesh.num_nodes - 1, -1, -1, dtype=int)
        elif strategy == "random":
            new_order = np.random.permutation(self.mesh.num_nodes)
        elif strategy == "spatial_x":
            new_order = np.argsort(self.mesh.node_coords[:, 0])
        else:
            raise NotImplementedError(f"renumber {strategy}")
        # apply permutation: new_order gives old indices in new order, so compute remap old->new
        remap = np.empty_like(new_order)
        remap[new_order] = np.arange(self.mesh.num_nodes, dtype=int)
        self.mesh.node_coords = self.mesh.node_coords[new_order]
        self.mesh.cell_connectivity = [
            list(remap[np.array(c)]) for c in self.mesh.cell_connectivity
        ]
        if (
            getattr(self.mesh, "boundary_faces_nodes", None) is not None
            and self.mesh.boundary_faces_nodes.size > 0
        ):
            self.mesh.boundary_faces_nodes = remap[self.mesh.boundary_faces_nodes]
        # clear derived fields
        self.mesh.cell_faces = []
        self.mesh.cell_neighbors = np.array([])
        self.mesh.cell_centroids = np.array([])
        self.mesh.face_midpoints = np.array([])
        self.mesh.face_normals = np.array([])
        self.mesh.face_areas = np.array([])
        self.mesh.cell_volumes = np.array([])

    def write_decompose_par_json_npy(self, output_dir: str, n_parts: int) -> None:
        """Write per-processor directories similar to OpenFOAM decomposePar output.

        For each partition p in 0..n_parts-1, create directory processor<p>/ and dump a
        numpy .npz file with the local mesh (node_coords, cells, boundary faces).
        """
        if self.elem_parts is None or len(self.elem_parts) != self.mesh.num_cells:
            raise RuntimeError(
                "Element partitioning not computed. Call partition_elements() first"
            )
        os.makedirs(output_dir, exist_ok=True)
        for p in range(n_parts):
            proc_dir = os.path.join(output_dir, f"processor{p}")
            os.makedirs(proc_dir, exist_ok=True)

            # select cells in partition
            mask = self.elem_parts == p
            local_cell_indices = np.nonzero(mask)[0]
            if local_cell_indices.size == 0:
                local_coords = np.empty(
                    (
                        0,
                        (
                            self.mesh.node_coords.shape[1]
                            if self.mesh.num_nodes > 0
                            else 3
                        ),
                    )
                )
                local_conn = np.empty((0,), dtype=object)
                unique_nodes = np.array([], dtype=int)
            else:
                local_cells = [
                    self.mesh.cell_connectivity[i] for i in local_cell_indices
                ]
                # flatten to node id list and get unique nodes
                all_nodes = (
                    np.hstack([np.array(c, dtype=int) for c in local_cells])
                    if len(local_cells) > 0
                    else np.array([], dtype=int)
                )
                unique_nodes = np.unique(all_nodes)
                local_node_map = {int(g): i for i, g in enumerate(unique_nodes)}
                local_coords = (
                    self.mesh.node_coords[unique_nodes]
                    if unique_nodes.size > 0
                    else np.empty((0, self.mesh.node_coords.shape[1]))
                )
                local_conn = [[local_node_map[int(g)] for g in c] for c in local_cells]
            # boundary faces
            if (
                getattr(self.mesh, "boundary_faces_nodes", None) is not None
                and self.mesh.boundary_faces_nodes.size > 0
            ):
                if local_cell_indices.size == 0:
                    is_in = np.zeros(
                        self.mesh.boundary_faces_nodes.shape[0], dtype=bool
                    )
                else:
                    local_node_set = set(unique_nodes.tolist())
                    is_in = np.array(
                        [
                            all(int(n) in local_node_set for n in face)
                            for face in self.mesh.boundary_faces_nodes
                        ],
                        dtype=bool,
                    )
                sel_idx = np.nonzero(is_in)[0]
                if sel_idx.size:
                    local_boundary_faces = np.array(
                        [
                            [
                                local_node_map[int(n)]
                                for n in self.mesh.boundary_faces_nodes[i]
                            ]
                            for i in sel_idx
                        ],
                        dtype=int,
                    )
                    local_boundary_tags = self.mesh.boundary_faces_tags[sel_idx]
                else:
                    local_boundary_faces = (
                        np.empty(
                            (0, self.mesh.boundary_faces_nodes.shape[1]), dtype=int
                        )
                        if self.mesh.boundary_faces_nodes.size
                        else np.empty((0, 0), dtype=int)
                    )
                    local_boundary_tags = np.empty((0,), dtype=int)
            else:
                local_boundary_faces = np.empty((0, 0), dtype=int)
                local_boundary_tags = np.empty((0,), dtype=int)
            # write files
            np.save(os.path.join(proc_dir, "node_coords.npy"), local_coords)
            np.save(os.path.join(proc_dir, "global_node_ids.npy"), unique_nodes)
            np.save(
                os.path.join(proc_dir, "cell_connectivity.npy"),
                np.array(local_conn, dtype=object),
            )
            np.save(
                os.path.join(proc_dir, "global_cell_indices.npy"), local_cell_indices
            )
            np.save(
                os.path.join(proc_dir, "boundary_faces_nodes.npy"), local_boundary_faces
            )
            np.save(
                os.path.join(proc_dir, "boundary_faces_tags.npy"), local_boundary_tags
            )
            meta = {
                "elem_part": int(p),
                "num_local_nodes": int(local_coords.shape[0]),
                "num_local_cells": int(local_cell_indices.size),
            }
            with open(os.path.join(proc_dir, "mesh.json"), "w") as fh:
                json.dump(meta, fh, indent=2)

    def reconstruct_par(self, decomposed_dir: str) -> Mesh:
        """Reconstruct a global mesh from processor*/local_mesh.npz outputs.

        This is a convenience function for testing the decomposition/reconstruction cycle.
        """
        proc_dirs = sorted(
            [d for d in os.listdir(decomposed_dir) if d.startswith("processor")]
        )
        node_blocks: List[Tuple[np.ndarray, np.ndarray]] = []
        global_cells: List[List[int]] = []
        for proc in proc_dirs:
            path = os.path.join(decomposed_dir, proc, "node_coords.npy")
            if not os.path.exists(path):
                continue
            gnode_ids = np.load(
                os.path.join(decomposed_dir, proc, "global_node_ids.npy")
            )
            local_coords = np.load(
                os.path.join(decomposed_dir, proc, "node_coords.npy")
            )
            local_conn = np.load(
                os.path.join(decomposed_dir, proc, "cell_connectivity.npy"),
                allow_pickle=True,
            )
            node_blocks.append((gnode_ids, local_coords))

            # convert local connectivity (local indices) to global ids
            for conn in local_conn:
                global_conn = [int(gnode_ids[int(li)]) for li in conn]
                global_cells.append(global_conn)
        if not node_blocks:
            raise RuntimeError("No processor data found")
        all_global = np.concatenate([b[0] for b in node_blocks])
        unique_global, inv = np.unique(all_global, return_inverse=True)
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

    def write_gmsh_per_processor(
        self, output_dir: str, n_parts: int, msh_filename: str = "processor{p}.msh"
    ) -> None:
        if self.elem_parts is None:
            raise RuntimeError("must partition before writing")
        for p in range(n_parts):
            proc_dir = os.path.join(output_dir, f"processor{p}")
            if not os.path.isdir(proc_dir):
                continue
            node_coords = np.load(os.path.join(proc_dir, "node_coords.npy"))
            cell_conn = np.load(
                os.path.join(proc_dir, "cell_connectivity.npy"), allow_pickle=True
            )
            out_path = os.path.join(proc_dir, msh_filename.format(p=p))
            with open(out_path, "w") as fh:
                fh.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
                fh.write(f"$Nodes\n{node_coords.shape[0]}\n")
                for i, xyz in enumerate(node_coords, start=1):
                    fh.write(f"{i} {float(xyz[0])} {float(xyz[1])} {float(xyz[2])}\n")
                fh.write("$EndNodes\n")
                fh.write(f"$Elements\n{len(cell_conn)}\n")
                eid = 1
                for conn in cell_conn:
                    nn = len(conn)
                    etype = 4 if nn == 4 else (2 if nn == 3 else 4)
                    nodes_str = " ".join(str(int(ci) + 1) for ci in conn)
                    fh.write(f"{eid} {etype} 0 {nodes_str}\n")
                    eid += 1
                fh.write("$EndElements\n")


# ----------------- Helpers -----------------


def mesh_print_summary(mesh: Mesh, brief: bool = True) -> None:
    print("--- Mesh summary ---")
    print(f"Dimension: {mesh.dimension}")
    print(f"Nodes: {mesh.num_nodes}; Cells: {mesh.num_cells}")
    if (
        mesh.num_cells > 0
        and getattr(mesh, "cell_volumes", None) is not None
        and mesh.cell_volumes.size > 0
    ):
        print(
            f"Cell volumes: min={float(np.min(mesh.cell_volumes)):.4g}, max={float(np.max(mesh.cell_volumes)):.4g}, mean={float(np.mean(mesh.cell_volumes)):.4g}"
        )
    if (
        getattr(mesh, "boundary_faces_nodes", None) is not None
        and mesh.boundary_faces_nodes.size > 0
    ):
        print(f"Boundary faces: {mesh.boundary_faces_nodes.shape[0]}")
        print(f"Boundary tag map: {mesh.boundary_tag_map}")
    if not brief:
        print("Sample node coords:", mesh.node_coords[:5])
        print("Sample cell connectivity:", mesh.cell_connectivity[:5])


def partition_print_summary(pm: PartitionManager) -> None:
    if pm.elem_parts is None:
        print("No partitioning computed yet.")
        return
    n_parts = int(np.max(pm.elem_parts) + 1)
    counts = np.bincount(pm.elem_parts, minlength=n_parts)
    print("--- Partition summary ---")
    print(f"Parts: {n_parts}")
    for p in range(n_parts):
        print(f" part {p}: cells = {counts[p]}")
    iface = 0
    if getattr(pm.mesh, "cell_faces", None) is not None:
        face_map = {}
        for ci, faces in enumerate(pm.mesh.cell_faces):
            for face in faces:
                key = tuple(sorted(face))
                face_map.setdefault(key, set()).add(int(pm.elem_parts[ci]))
        iface = sum(1 for s in face_map.values() if len(s) > 1)
    print(f"Estimated inter-part interface faces: {iface}")


def build_halo_indices_from_decomposed(decomposed_dir: str):
    proc_dirs = sorted(
        [d for d in os.listdir(decomposed_dir) if d.startswith("processor")]
    )
    rank_data = {}
    for proc in proc_dirs:
        p = int(proc.replace("processor", ""))
        proc_path = os.path.join(decomposed_dir, proc)
        gnode_ids = np.load(os.path.join(proc_path, "global_node_ids.npy"))
        rank_data[p] = {"global_node_ids": gnode_ids, "proc_path": proc_path}
    node_to_ranks = {}
    for rank, info in rank_data.items():
        for gid in info["global_node_ids"]:
            node_to_ranks.setdefault(int(gid), []).append(rank)
    out = {}
    for rank, info in rank_data.items():
        gnodes = np.array(info["global_node_ids"], dtype=int)
        local_to_global = gnodes.copy()
        global_to_local = {int(g): int(i) for i, g in enumerate(gnodes)}
        neighbor_ranks = set()
        for g in gnodes:
            for r in node_to_ranks.get(int(g), []):
                if r != rank:
                    neighbor_ranks.add(r)
        neighbors = {}
        for nbr in sorted(neighbor_ranks):
            overlap = [int(g) for g in gnodes if nbr in node_to_ranks.get(int(g), [])]
            send_local = [global_to_local[g] for g in overlap]
            neighbors[nbr] = {
                "send_local_indices": send_local,
                "send_global_ids": overlap,
            }
        out[rank] = {
            "local_to_global": local_to_global,
            "global_to_local": global_to_local,
            "neighbors": neighbors,
        }
    return out
