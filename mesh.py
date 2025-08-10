"""
Refactored mesh handling and partitioning for parallel FVM solver.

Key classes:
- Mesh: read + analyze geometry/connectivity. Independent of partitioning/renumbering.
- PartitionManager: partition elements (using METIS), perform renumbering, and write per-processor folders
  following an OpenFOAM-like decomposePar pattern (creates processorN/ with a .npz mesh dump).

Usage example (at bottom): read mesh -> analyze_mesh() -> PartitionManager.decompose_par(...)
"""

import os
import json
from typing import Dict, List, Tuple, Any, Optional

import gmsh
import numpy as np

# Optional import for partitioning; error only raised if partition requested
try:
    import metis
except Exception:  # pragma: no cover - handled below
    metis = None


class Mesh:
    """Represents a geometric mesh and computes FVM geometric/connectivity data.

    Responsibilities:
    - read mesh from gmsh (.msh)
    - compute cell centroids, faces, neighbors, face areas/normals, cell volumes
    - provide mesh_data export for solvers

    Note: This class DOES NOT perform partitioning or node renumbering. Those are
    implemented in PartitionManager.
    """

    def __init__(self) -> None:
        # core mesh attributes
        self.dimension: int = 0
        self.num_nodes: int = 0
        self.num_cells: int = 0

        # raw gmsh tags -> zero-based indices mapping
        self._gmsh_node_tags: np.ndarray = np.array([])  # original gmsh tags
        self._tag_to_index: Dict[int, int] = {}

        # geometry/connectivity
        self.node_coords: np.ndarray = np.array([])  # shape (N,3)
        self.cell_connectivity: List[List[int]] = (
            []
        )  # list of node index lists per cell
        self.cell_type_ids: np.ndarray = np.array([])  # integer type id per cell
        self.cell_type_map: Dict[int, Dict[str, Any]] = {}

        # boundary info
        self.boundary_faces_nodes: np.ndarray = np.array([])  # (M, face_nodes)
        self.boundary_faces_tags: np.ndarray = np.array([])
        self.boundary_tag_map: Dict[str, int] = {}

        # computed FVM data (populated by analyze_mesh)
        self.cell_centroids: np.ndarray = np.array([])  # (num_cells, 3)
        self.cell_volumes: np.ndarray = np.array([])  # (num_cells,)
        self.cell_faces: List[List[List[int]]] = (
            []
        )  # cell_faces[cell_idx] = [face_node_list,...]
        self.face_midpoints: np.ndarray = np.array([])  # (num_cells, max_faces, 3)
        self.face_normals: np.ndarray = np.array([])  # (num_cells, max_faces, 3)
        self.face_areas: np.ndarray = np.array([])  # (num_cells, max_faces)
        self.cell_neighbors: np.ndarray = np.array([])  # (num_cells, max_faces)

    # ------------------- I/O -------------------
    def read_gmsh(self, msh_file: str, gmsh_verbose: int = 0) -> None:
        """Read mesh nodes and elements from Gmsh .msh file.

        This populates node_coords, cell_connectivity, cell_type_map, and
        boundary face lists (if physical groups exist).
        """
        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", gmsh_verbose)
        try:
            gmsh.open(msh_file)

            raw_tags, raw_coords, _ = gmsh.model.mesh.getNodes()
            coords = np.array(raw_coords).reshape(-1, 3)

            self._gmsh_node_tags = np.array(raw_tags)
            self._tag_to_index = {int(tag): i for i, tag in enumerate(raw_tags)}
            self.node_coords = coords
            self.num_nodes = coords.shape[0]

            elem_types, elem_tags_list, connectivity_list = (
                gmsh.model.mesh.getElements()
            )

            # determine dimensionality from element properties (1/2/3D)
            dim = 0
            for et in elem_types:
                props = gmsh.model.mesh.getElementProperties(et)
                dim = max(dim, int(props[1]))
            self.dimension = dim

            # collect elements of highest dimension (cells)
            all_cell_tags: List[int] = []
            all_cell_conn: List[List[int]] = []
            all_type_ids: List[int] = []
            type_counter = 0

            for i, et in enumerate(elem_types):
                props = gmsh.model.mesh.getElementProperties(et)
                et_dim = int(props[1])
                n_nodes_et = int(props[3])
                if et_dim != self.dimension:
                    continue
                # store type info
                self.cell_type_map[type_counter] = {
                    "name": props[0],
                    "num_nodes": n_nodes_et,
                }

                raw_tags = elem_tags_list[i]
                raw_conn = np.array(connectivity_list[i]).reshape(-1, n_nodes_et)
                # convert gmsh node tags to zero-based indices
                mapped_conn = np.vectorize(self._tag_to_index.get)(raw_conn)

                all_cell_tags.extend(raw_tags)
                all_cell_conn.extend(mapped_conn.tolist())
                all_type_ids.extend([type_counter] * len(raw_tags))
                type_counter += 1

            if all_cell_conn:
                self.cell_connectivity = all_cell_conn
                self.cell_type_ids = np.array(all_type_ids, dtype=int)
                self.num_cells = len(self.cell_connectivity)

            # boundary / physical groups
            self._read_gmsh_boundary_groups()

        finally:
            gmsh.finalize()

    def _read_gmsh_boundary_groups(self) -> None:
        """Read physical groups at dimension (self.dimension - 1) as boundary faces."""
        bdim = self.dimension - 1
        if bdim < 0:
            return

        all_faces = []
        all_tags = []
        phys_groups = gmsh.model.getPhysicalGroups(dim=bdim)
        for dim, tag in phys_groups:
            name = gmsh.model.getPhysicalName(dim, tag)
            self.boundary_tag_map[name] = tag
            entities = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
            for ent in entities:
                etypes, etags_list, etconn_list = gmsh.model.mesh.getElements(dim, ent)
                if len(etconn_list) == 0:
                    continue
                # assume first entry corresponds to faces/type
                n_nodes_face = gmsh.model.mesh.getElementProperties(etypes[0])[3]
                faces = np.array(etconn_list[0]).reshape(-1, n_nodes_face)
                faces_idx = np.vectorize(self._tag_to_index.get)(faces)
                all_faces.append(faces_idx)
                all_tags.extend([tag] * faces_idx.shape[0])

        if all_faces:
            self.boundary_faces_nodes = np.vstack(all_faces)
            self.boundary_faces_tags = np.array(all_tags)

    # ------------------- Analysis (independent of partitioning) -------------------
    def analyze_mesh(self) -> None:
        """Compute centroids, faces, neighbors, face areas/normals, and cell volumes.

        This method is independent of any renumbering or partitioning and can be
        called at any time after the mesh is loaded.
        """
        if self.num_cells == 0:
            raise RuntimeError("No cells available. Call read_gmsh() first.")

        self._compute_cell_centroids()
        self._extract_cell_faces()
        self._compute_cell_neighbors()
        self._compute_face_midpoints_areas_normals()
        self._compute_cell_volumes()

    def _compute_cell_centroids(self) -> None:
        self.cell_centroids = np.array(
            [np.mean(self.node_coords[c], axis=0) for c in self.cell_connectivity]
        )

    def _extract_cell_faces(self) -> None:
        # face definitions for some common element node counts
        face_templates = {
            4: [[0, 1, 2], [0, 3, 1], [1, 3, 2], [2, 0, 3]],  # tet
            8: [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [0, 1, 5, 4],
                [1, 2, 6, 5],
                [2, 3, 7, 6],
                [3, 0, 4, 7],
            ],  # hex
            6: [
                [0, 1, 2],
                [3, 4, 5],
                [0, 1, 4, 3],
                [1, 2, 5, 4],
                [2, 0, 3, 5],
            ],  # wedge
        }

        self.cell_faces = []
        for conn in self.cell_connectivity:
            n_nodes = len(conn)
            if self.dimension == 2:
                faces = [[conn[i], conn[(i + 1) % n_nodes]] for i in range(n_nodes)]
            elif self.dimension == 3:
                if n_nodes not in face_templates:
                    raise NotImplementedError(
                        f"3D element with {n_nodes} nodes not supported yet"
                    )
                faces = [
                    [conn[idx] for idx in face] for face in face_templates[n_nodes]
                ]
            else:
                faces = []
            self.cell_faces.append(faces)

    def _compute_cell_neighbors(self) -> None:
        max_faces = max(len(f) for f in self.cell_faces)
        num_cells = self.num_cells
        neighbors = -np.ones((num_cells, max_faces), dtype=int)
        face_map: Dict[Tuple[int, ...], List[int]] = {}

        for ci, faces in enumerate(self.cell_faces):
            for face in faces:
                key = tuple(sorted(face))
                face_map.setdefault(key, []).append(ci)

        for ci, faces in enumerate(self.cell_faces):
            for fi, face in enumerate(faces):
                key = tuple(sorted(face))
                elems = face_map.get(key, [])
                if len(elems) == 2:
                    neighbors[ci, fi] = elems[0] if elems[1] == ci else elems[1]
                else:
                    neighbors[ci, fi] = -1

        self.cell_neighbors = neighbors

    def _compute_face_midpoints_areas_normals(self) -> None:
        max_faces = self.cell_neighbors.shape[1]
        num_cells = self.num_cells
        self.face_midpoints = np.zeros((num_cells, max_faces, 3))
        self.face_normals = np.zeros((num_cells, max_faces, 3))
        self.face_areas = np.zeros((num_cells, max_faces))

        for ci, faces in enumerate(self.cell_faces):
            for fi, face_nodes in enumerate(faces):
                nodes = self.node_coords[face_nodes]
                self.face_midpoints[ci, fi] = np.mean(nodes, axis=0)
                if self.dimension == 2:
                    p0, p1 = nodes[0], nodes[1]
                    edge = p1 - p0
                    length = np.linalg.norm(edge)
                    self.face_areas[ci, fi] = length
                    if length > 0:
                        # outward normal in-plane (z component zero)
                        n = np.array([edge[1], -edge[0], 0.0]) / length
                        self.face_normals[ci, fi] = n
                elif self.dimension == 3:
                    # compute polygon normal by triangulation from centroid
                    centroid = np.mean(nodes, axis=0)
                    area_vec = np.zeros(3)
                    for k in range(len(nodes)):
                        p1 = nodes[k]
                        p2 = nodes[(k + 1) % len(nodes)]
                        area_vec += np.cross(p1 - centroid, p2 - centroid)
                    area_vec *= 0.5
                    area = np.linalg.norm(area_vec)
                    self.face_areas[ci, fi] = area
                    if area > 0:
                        self.face_normals[ci, fi] = area_vec / area

        # orient outward
        for ci in range(num_cells):
            for fi in range(len(self.cell_faces[ci])):
                vec = self.face_midpoints[ci, fi] - self.cell_centroids[ci]
                if np.dot(self.face_normals[ci, fi], vec) < 0:
                    self.face_normals[ci, fi] *= -1

    def _compute_cell_volumes(self) -> None:
        self.cell_volumes = np.zeros(self.num_cells)
        if self.dimension == 2:
            for i, conn in enumerate(self.cell_connectivity):
                nodes = self.node_coords[conn][:, :2]
                x = nodes[:, 0]
                y = nodes[:, 1]
                self.cell_volumes[i] = 0.5 * abs(
                    np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))
                )
        elif self.dimension == 3:
            # volume from faces (Divergence theorem)
            contrib = (self.face_midpoints * self.face_normals).sum(
                axis=2
            ) * self.face_areas
            self.cell_volumes = np.sum(contrib, axis=1) / 3.0
        else:
            for i, conn in enumerate(self.cell_connectivity):
                p0, p1 = self.node_coords[conn]
                self.cell_volumes[i] = np.linalg.norm(p1 - p0)

    # ------------------- Export -------------------
    def get_mesh_data(self) -> Dict[str, Any]:
        """Return structured data useful for FVM solver or partition writer."""
        return {
            "dimension": int(self.dimension),
            "node_coords": self.node_coords,
            "cell_connectivity": self.cell_connectivity,
            "cell_type_ids": self.cell_type_ids,
            "cell_volumes": self.cell_volumes,
            "cell_centroids": self.cell_centroids,
            "face_areas": self.face_areas,
            "face_normals": self.face_normals,
            "cell_neighbors": self.cell_neighbors,
            "boundary_faces_nodes": self.boundary_faces_nodes,
            "boundary_faces_tags": self.boundary_faces_tags,
            "boundary_tag_map": self.boundary_tag_map,
        }


class PartitionManager:
    """Manage element partitioning and renumbering. Produces OpenFOAM-like processor folders.

    Responsibilities:
    - compute element partitioning using METIS
    - optional node renumbering per partition
    - write per-processor mesh dumps into `processor{n}` directories
    """

    def __init__(self, mesh: Mesh):
        self.mesh = mesh
        self.elem_parts: Optional[np.ndarray] = None

    def partition_elements(self, n_parts: int, method: str = "metis") -> np.ndarray:
        """Partition elements into n_parts. Returns array elem_parts of length num_cells.

        method currently supports 'metis'. ValueError if metis not available.
        """
        if n_parts <= 1:
            self.elem_parts = np.zeros(self.mesh.num_cells, dtype=int)
            return self.elem_parts

        if method == "metis":
            if metis is None:
                raise ImportError(
                    "METIS python bindings not available. Install the 'metis' package."
                )

            # Build adjacency list from mesh.cell_neighbors
            if self.mesh.cell_neighbors.size == 0:
                self.mesh._compute_cell_centroids()
                self.mesh._extract_cell_faces()
                self.mesh._compute_cell_neighbors()

            adjacency: List[List[int]] = [[] for _ in range(self.mesh.num_cells)]
            for i in range(self.mesh.num_cells):
                for nb in self.mesh.cell_neighbors[i]:
                    if nb != -1:
                        adjacency[i].append(int(nb))

            _, parts = metis.part_graph(adjacency, nparts=n_parts)
            self.elem_parts = np.array(parts, dtype=int)
            return self.elem_parts
        else:
            raise NotImplementedError(f"Partition method '{method}' not implemented")

    def renumber_nodes_global(self, strategy: str = "sequential") -> None:
        """A simple global node renumbering that reorders nodes in mesh.node_coords.

        This is separate from partitioning and may be called independently.
        Strategies: 'sequential', 'reverse', 'spatial_x', 'random'.
        """
        if self.mesh.num_nodes == 0:
            return
        if strategy == "sequential":
            new_order = np.arange(self.mesh.num_nodes)
        elif strategy == "reverse":
            new_order = np.arange(self.mesh.num_nodes - 1, -1, -1)
        elif strategy == "random":
            new_order = np.random.permutation(self.mesh.num_nodes)
        elif strategy == "spatial_x":
            new_order = np.argsort(self.mesh.node_coords[:, 0])
        else:
            raise NotImplementedError(f"Renumber strategy '{strategy}' not implemented")

        # apply permutation
        remap = np.empty_like(new_order)
        remap[new_order] = np.arange(self.mesh.num_nodes)
        self.mesh.node_coords = self.mesh.node_coords[new_order]
        self.mesh.cell_connectivity = [
            list(remap[np.array(c)]) for c in self.mesh.cell_connectivity
        ]
        if self.mesh.boundary_faces_nodes.size > 0:
            self.mesh.boundary_faces_nodes = remap[self.mesh.boundary_faces_nodes]

        # Mark computed fields stale
        self.mesh.cell_faces = []
        self.mesh.cell_neighbors = np.array([])

    def write_decompose_par(
        self, output_dir: str, n_parts: int, include_face_info: bool = True
    ) -> None:
        """Write per-processor directories similar to OpenFOAM decomposePar output.

        For each partition p in 0..n_parts-1, create directory processor<p>/ and dump a
        numpy .npz file with the local mesh (node_coords, cells, boundary faces).
        """
        if self.elem_parts is None or len(self.elem_parts) != self.mesh.num_cells:
            raise RuntimeError(
                "Element partitioning not computed. Call partition_elements() first."
            )

        os.makedirs(output_dir, exist_ok=True)

        for p in range(n_parts):
            proc_dir = os.path.join(output_dir, f"processor{p}")
            os.makedirs(proc_dir, exist_ok=True)

            # select cells in partition
            mask = self.elem_parts == p
            local_cell_indices = np.nonzero(mask)[0]
            local_cells = [self.mesh.cell_connectivity[i] for i in local_cell_indices]

            # determine local node set and create local-to-global map
            unique_nodes = np.unique(np.concatenate(local_cells).ravel())
            local_node_map = {int(g): i for i, g in enumerate(unique_nodes)}
            local_coords = self.mesh.node_coords[unique_nodes]

            # remap local cell connectivity to local node indices
            local_conn = [[local_node_map[int(g)] for g in c] for c in local_cells]

            # extract boundary faces that belong to this partition (if any)
            local_boundary_faces = None
            local_boundary_tags = None
            if self.mesh.boundary_faces_nodes.size > 0:
                # a boundary face belongs to this partition if all its nodes are in unique_nodes
                is_in = [
                    all(int(n) in local_node_map for n in face)
                    for face in self.mesh.boundary_faces_nodes
                ]
                if any(is_in):
                    sel = np.nonzero(np.array(is_in))[0]
                    local_boundary_faces = np.array(
                        [
                            [
                                local_node_map[int(n)]
                                for n in self.mesh.boundary_faces_nodes[i]
                            ]
                            for i in sel
                        ]
                    )
                    local_boundary_tags = self.mesh.boundary_faces_tags[sel]

            save_path = os.path.join(proc_dir, "local_mesh.npz")
            # Ensure arrays are not None for np.savez
            if local_boundary_faces is None:
                local_boundary_faces = (
                    np.empty((0, self.mesh.boundary_faces_nodes.shape[1]), dtype=int)
                    if self.mesh.boundary_faces_nodes.size > 0
                    else np.empty((0, 0), dtype=int)
                )
            if local_boundary_tags is None:
                local_boundary_tags = (
                    np.empty((0,), dtype=self.mesh.boundary_faces_tags.dtype)
                    if self.mesh.boundary_faces_tags.size > 0
                    else np.empty((0,), dtype=int)
                )
            # Save arrays and python objects; use allow_pickle via np.savez
            np.savez(
                save_path,
                node_coords=local_coords,
                cell_connectivity=np.array(local_conn, dtype=object),
                global_cell_indices=local_cell_indices,
                global_node_ids=unique_nodes,
                boundary_faces_nodes=local_boundary_faces,
                boundary_faces_tags=local_boundary_tags,
                elem_part=p,
            )

    def reconstruct_par(self, decomposed_dir: str) -> Mesh:
        """Reconstruct a global mesh from processor*/local_mesh.npz outputs.

        This is a convenience function for testing the decomposition/reconstruction cycle.
        """
        proc_dirs = [d for d in os.listdir(decomposed_dir) if d.startswith("processor")]
        proc_dirs = sorted(proc_dirs)

        global_nodes_list = []
        global_cell_conn = []
        for proc in proc_dirs:
            path = os.path.join(decomposed_dir, proc, "local_mesh.npz")
            if not os.path.exists(path):
                continue
            data = np.load(path, allow_pickle=True)
            gnode_ids = data["global_node_ids"]
            local_coords = data["node_coords"]
            local_conn = data["cell_connectivity"]

            # append nodes
            global_nodes_list.append((gnode_ids, local_coords))
            # remap local connectivity to global node id (already global ids exist)
            for conn in local_conn:
                # conn holds local indices; convert to global ids then append
                global_conn = [int(gnode_ids[int(li)]) for li in conn]
                global_cell_conn.append(global_conn)

        # build global node array by placing coordinates into correct positions
        all_global_ids = np.concatenate([t[0] for t in global_nodes_list])
        unique_global_ids, inv = np.unique(all_global_ids, return_inverse=True)
        max_id = unique_global_ids.max()
        # create array large enough, fill sparse and compress
        coords_shape = global_nodes_list[0][1].shape[1]
        global_coords = np.zeros((len(unique_global_ids), coords_shape))
        pos = 0
        for g_ids, coords in global_nodes_list:
            for i, gid in enumerate(g_ids):
                idx = np.where(unique_global_ids == gid)[0][0]
                global_coords[idx, :] = coords[i]

        # create Mesh and populate
        new_mesh = Mesh()
        new_mesh.node_coords = global_coords
        new_mesh.num_nodes = global_coords.shape[0]
        new_mesh.cell_connectivity = global_cell_conn
        new_mesh.num_cells = len(global_cell_conn)
        # run analysis to compute derived quantities
        new_mesh.analyze_mesh()
        return new_mesh


# ------------------- Example usage -------------------
if __name__ == "__main__":
    # Example run: read mesh, analyze, partition and write processor folders
    msh_path = "./data/river_mixed.msh"
    out_dir = "./decomposed"
    nprocs = 4

    mesh = Mesh()
    mesh.read_gmsh(msh_path)
    mesh.analyze_mesh()
    mesh_summary = mesh.get_mesh_data()
    print("Loaded mesh with", mesh.num_nodes, "nodes and", mesh.num_cells, "cells")

    pm = PartitionManager(mesh)
    parts = pm.partition_elements(nprocs)
    print("Partition counts:", np.bincount(parts))

    pm.write_decompose_par(out_dir, nprocs)
    print(f"Wrote decomposition into {out_dir}/processor* directories")
