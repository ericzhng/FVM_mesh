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

        # geometry/connectivity
        self.node_coords: np.ndarray = np.array([])  # (N,3)
        self.cell_connectivity: List[List[int]] = (
            []
        )  # list of node index lists per cell
        self.cell_type_ids: np.ndarray = np.array([])
        self.cell_type_map: Dict[int, Dict[str, Any]] = {}

        # boundary info
        self.boundary_faces_nodes: np.ndarray = np.array([])  # (M, n_face_nodes)
        self.boundary_faces_tags: np.ndarray = np.array([])
        self.boundary_tag_map: Dict[str, int] = {}

        # computed fields
        self.cell_centroids: np.ndarray = np.array([])
        self.cell_volumes: np.ndarray = np.array([])
        self.cell_faces: List[List[List[int]]] = []
        self.face_midpoints: np.ndarray = np.array([])
        self.face_normals: np.ndarray = np.array([])
        self.face_areas: np.ndarray = np.array([])
        self.cell_neighbors: np.ndarray = np.array([])

    # ---------- I/O ----------
    def read_gmsh(self, msh_file: str, gmsh_verbose: int = 0) -> None:
        """Read mesh nodes and elements from Gmsh .msh file.

        This populates node_coords, cell_connectivity, cell_type_map, and
        boundary face lists (if physical groups exist).
        """
        if gmsh is None:
            raise RuntimeError("gmsh python API is not available in this environment")
        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", gmsh_verbose)
        try:
            gmsh.open(msh_file)
            raw_tags, raw_coords, _ = gmsh.model.mesh.getNodes()
            coords = np.array(raw_coords).reshape(-1, 3)
            self.node_coords = coords
            self.num_nodes = coords.shape[0]
            self._gmsh_node_tags = np.array(raw_tags)
            self._tag_to_index = {int(t): i for i, t in enumerate(raw_tags)}

            elem_types, elem_tags_list, connectivity_list = (
                gmsh.model.mesh.getElements()
            )

            # determine mesh dimension
            dim = 0
            for et in elem_types:
                props = gmsh.model.mesh.getElementProperties(et)
                dim = max(dim, int(props[1]))
            self.dimension = dim

            all_cell_conn: List[List[int]] = []
            all_type_ids: List[int] = []
            type_counter = 0
            for i, et in enumerate(elem_types):
                props = gmsh.model.mesh.getElementProperties(et)
                et_dim = int(props[1])
                n_nodes_et = int(props[3])
                if et_dim != self.dimension:
                    continue
                # store type info (name, number of nodes)
                self.cell_type_map[type_counter] = {
                    "name": props[0],
                    "num_nodes": n_nodes_et,
                }
                raw_conn = np.array(connectivity_list[i]).reshape(-1, n_nodes_et)
                try:
                    mapped_conn = np.vectorize(lambda x: self._tag_to_index[int(x)])(
                        raw_conn
                    )
                except KeyError as ex:
                    raise KeyError(f"gmsh node tag missing: {ex}")
                all_cell_conn.extend(mapped_conn.tolist())
                all_type_ids.extend([type_counter] * mapped_conn.shape[0])
                type_counter += 1
            if all_cell_conn:
                self.cell_connectivity = all_cell_conn
                self.cell_type_ids = np.array(all_type_ids, dtype=int)
                self.num_cells = len(self.cell_connectivity)

            # read boundary physical groups
            self._read_gmsh_boundary_groups()
        finally:
            gmsh.finalize()

    def _read_gmsh_boundary_groups(self) -> None:
        """Read physical groups at dimension (self.dimension - 1) as boundary faces.

        Notes:
        - We filter physical groups by desired boundary dimension to avoid relying on API
          providing a 'dim' argument.
        - If physical groups contain faces of different element node counts we currently
          store them in separate contiguous blocks (vstack requires consistent number of
          nodes per face type) — mixed face types across groups may not vstack.
        """
        if gmsh is None:
            return
        bdim = self.dimension - 1
        if bdim < 0:
            return
        all_faces = []
        all_tags: List[int] = []
        phys = gmsh.model.getPhysicalGroups()
        phys = [pg for pg in phys if pg[0] == bdim]
        for dim, tag in phys:
            try:
                name = gmsh.model.getPhysicalName(dim, tag)
            except Exception:
                name = str(tag)
            self.boundary_tag_map[name] = tag
            entities = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
            for ent in entities:
                etypes, etags_list, etconn_list = gmsh.model.mesh.getElements(dim, ent)
                if len(etconn_list) == 0:
                    continue
                n_nodes = gmsh.model.mesh.getElementProperties(etypes[0])[3]
                faces = np.array(etconn_list[0]).reshape(-1, n_nodes)
                try:
                    faces_idx = np.vectorize(lambda x: self._tag_to_index[int(x)])(
                        faces
                    )
                except KeyError as ex:
                    raise KeyError(f"Boundary face node tag not found: {ex}")
                all_faces.append(faces_idx)
                all_tags.extend([tag] * faces_idx.shape[0])
        if all_faces:
            # Ensure all faces have same node count before vstacking
            lengths = [a.shape[1] for a in all_faces]
            if len(set(lengths)) != 1:
                face_list = []
                for arr in all_faces:
                    for r in arr:
                        face_list.append(r)
                self.boundary_faces_nodes = np.array(face_list, dtype=int)
                self.boundary_faces_tags = np.array(all_tags, dtype=int)
            else:
                self.boundary_faces_nodes = np.vstack(all_faces)
                self.boundary_faces_tags = np.array(all_tags, dtype=int)

    # ---------- Analysis ----------
    def analyze_mesh(self) -> None:
        """Compute centroids, faces, neighbors, face areas/normals, and cell volumes.

        This method is independent of any renumbering or partitioning and can be
        called at any time after the mesh is loaded.
        """
        if self.num_cells == 0:
            raise RuntimeError(
                "No cells available. Call read_gmsh or populate cells first"
            )
        self._compute_centroids()
        self._extract_cell_faces()
        self._compute_cell_neighbors()
        self._compute_face_midpoints_areas_normals()
        self._compute_cell_volumes()

    def _compute_centroids(self) -> None:
        self.cell_centroids = np.array(
            [np.mean(self.node_coords[c], axis=0) for c in self.cell_connectivity]
        )

    def _extract_cell_faces(self) -> None:
        # face templates for common element node counts (3D); 2D handled generically
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
            n = len(conn)
            if self.dimension == 2:
                faces = [[conn[i], conn[(i + 1) % n]] for i in range(n)]
            elif self.dimension == 3:
                if n not in face_templates:
                    raise NotImplementedError(
                        f"3D element with {n} nodes not supported"
                    )
                faces = [[conn[idx] for idx in face] for face in face_templates[n]]
            else:
                faces = []
            self.cell_faces.append(faces)

    def _compute_cell_neighbors(self) -> None:
        if not self.cell_faces:
            self.cell_neighbors = np.array([])
            return
        max_faces = max(len(f) for f in self.cell_faces)
        num = self.num_cells
        neighbors = -np.ones((num, max_faces), dtype=int)
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
        if self.cell_neighbors.size == 0:
            self.face_midpoints = np.array([])
            self.face_normals = np.array([])
            self.face_areas = np.array([])
            return
        max_faces = self.cell_neighbors.shape[1]
        num = self.num_cells
        self.face_midpoints = np.zeros((num, max_faces, 3))
        self.face_normals = np.zeros((num, max_faces, 3))
        self.face_areas = np.zeros((num, max_faces))
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
        # orient outward (face normal should point away from cell centroid)
        for ci in range(num):
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
            contrib = (self.face_midpoints * self.face_normals).sum(
                axis=2
            ) * self.face_areas
            self.cell_volumes = np.sum(contrib, axis=1) / 3.0
        else:
            # Unknown geometric type: set zero volumes
            self.cell_volumes = np.zeros(self.num_cells)

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


# If module executed directly, do nothing. Unit tests exist separately.
if __name__ == "__main__":
    print("mesh.py module. Import classes and use in your script.")
