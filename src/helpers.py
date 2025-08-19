import os
import numpy as np

from src.mesh_analysis import Mesh
from src.mesh_partition import PartitionManager


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
