import numpy as np
from typing import Dict, List, Tuple, Any

import gmsh


class Mesh2D:
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
        self.cell_connectivity: List[List[int]] = []
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

        # Quality metrics
        self.min_max_volume_ratio: float = 0.0
        self.cell_skewness_values: np.ndarray = np.array([])
        self.cell_non_orthogonality_values: np.ndarray = np.array([])
        self.cell_aspect_ratio_values: np.ndarray = np.array([])
        self.connectivity_issues: List[str] = []

        # Analysis flag
        self._is_analyzed: bool = False  # New flag

        # Internal gmsh node tag to index mapping
        self._tag_to_index: Dict[int, int] = {}

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
            # self._gmsh_node_tags = np.array(raw_tags) # Not strictly needed as an attribute
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
        self._is_analyzed = True  # Set flag after successful analysis

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

    def compute_quality(self) -> None:
        """Computes mesh quality metrics and stores them as attributes."""
        if not self._is_analyzed:
            raise RuntimeError("Mesh must be analyzed before computing quality.")

        # Initialize quality metrics
        self.min_max_volume_ratio = 0.0
        self.cell_skewness_values = np.zeros(self.num_cells)
        self.cell_non_orthogonality_values = np.zeros(self.num_cells)
        self.cell_aspect_ratio_values = np.zeros(self.num_cells)
        self.connectivity_issues = []

        if self.num_cells == 0:
            return

        # 1. Ratio of smallest to biggest cell volume
        min_volume = np.min(self.cell_volumes)
        max_volume = np.max(self.cell_volumes)
        if max_volume > 0:
            self.min_max_volume_ratio = min_volume / max_volume
        else:
            self.min_max_volume_ratio = 0.0

        # Ensure cell_faces is populated for non-orthogonality and aspect ratio
        if not self.cell_faces:
            self._extract_cell_faces()  # Re-extract if not already done

        # 2. Skewness, Non-Orthogonality, and Aspect Ratio (2D only for now)
        if self.dimension == 2:
            for i, conn in enumerate(self.cell_connectivity):
                nodes = self.node_coords[conn][:, :2]
                num_nodes_in_cell = len(nodes)

                # Skewness (angle-based)
                if num_nodes_in_cell == 3:  # Triangle
                    v0 = nodes[1] - nodes[0]
                    v1 = nodes[2] - nodes[1]
                    v2 = nodes[0] - nodes[2]
                    l0 = np.linalg.norm(v0)
                    l1 = np.linalg.norm(v1)
                    l2 = np.linalg.norm(v2)

                    if l0 > 0 and l1 > 0 and l2 > 0:
                        angle0 = np.degrees(np.arccos(np.dot(-v2, v0) / (l2 * l0)))
                        angle1 = np.degrees(np.arccos(np.dot(-v0, v1) / (l0 * l1)))
                        angle2 = np.degrees(np.arccos(np.dot(-v1, v2) / (l1 * l2)))
                        angles = np.array([angle0, angle1, angle2])
                        self.cell_skewness_values[i] = np.max(np.abs(angles - 60)) / 60
                    else:
                        self.cell_skewness_values[i] = 1.0  # Degenerate triangle

                    # Aspect Ratio (longest edge / shortest edge)
                    edge_lengths = np.array([l0, l1, l2])
                    if np.min(edge_lengths) > 0:
                        self.cell_aspect_ratio_values[i] = np.max(
                            edge_lengths
                        ) / np.min(edge_lengths)
                    else:
                        self.cell_aspect_ratio_values[i] = np.inf  # Degenerate

                elif num_nodes_in_cell == 4:  # Quadrilateral
                    # Skewness (angle-based)
                    angles = []
                    for j in range(num_nodes_in_cell):
                        p_prev = nodes[(j - 1 + num_nodes_in_cell) % num_nodes_in_cell]
                        p_curr = nodes[j]
                        p_next = nodes[(j + 1) % num_nodes_in_cell]

                        v1 = p_prev - p_curr
                        v2 = p_next - p_curr

                        l1 = np.linalg.norm(v1)
                        l2 = np.linalg.norm(v2)

                        if l1 > 0 and l2 > 0:
                            dot_product = np.dot(v1, v2)
                            angle = np.degrees(
                                np.arccos(np.clip(dot_product / (l1 * l2), -1.0, 1.0))
                            )
                            angles.append(angle)
                        else:
                            angles.append(180.0)  # Degenerate edge

                    if angles:
                        self.cell_skewness_values[i] = (
                            np.max(np.abs(np.array(angles) - 90)) / 90
                        )
                    else:
                        self.cell_skewness_values[i] = 1.0  # Degenerate quad

                    # Aspect Ratio (longest edge / shortest edge)
                    edge_lengths = []
                    for j in range(num_nodes_in_cell):
                        edge_lengths.append(
                            np.linalg.norm(
                                nodes[j] - nodes[(j + 1) % num_nodes_in_cell]
                            )
                        )
                    edge_lengths = np.array(edge_lengths)
                    if np.min(edge_lengths) > 0:
                        self.cell_aspect_ratio_values[i] = np.max(
                            edge_lengths
                        ) / np.min(edge_lengths)
                    else:
                        self.cell_aspect_ratio_values[i] = np.inf  # Degenerate

                else:
                    # For other 2D cell types, skewness/aspect ratio not computed
                    self.cell_skewness_values[i] = 0.0
                    self.cell_aspect_ratio_values[i] = 0.0

            # Non-Orthogonality (2D)
            # Angle between cell centroid vector and face normal
            for ci in range(self.num_cells):
                cell_centroid = self.cell_centroids[ci]
                max_non_orthogonality = 0.0
                for fi, face_nodes in enumerate(self.cell_faces[ci]):
                    if fi < len(self.face_midpoints[ci]) and fi < len(
                        self.face_normals[ci]
                    ):
                        face_midpoint = self.face_midpoints[ci, fi]
                        face_normal = self.face_normals[ci, fi]

                        # Vector from cell centroid to face midpoint
                        centroid_to_face_vec = face_midpoint - cell_centroid
                        norm_centroid_to_face_vec = np.linalg.norm(centroid_to_face_vec)
                        norm_face_normal = np.linalg.norm(face_normal)

                        if norm_centroid_to_face_vec > 0 and norm_face_normal > 0:
                            dot_product = np.dot(centroid_to_face_vec, face_normal)
                            # Clip to avoid floating point errors causing arccos of > 1 or < -1
                            angle_rad = np.arccos(
                                np.clip(
                                    dot_product
                                    / (norm_centroid_to_face_vec * norm_face_normal),
                                    -1.0,
                                    1.0,
                                )
                            )
                            angle_deg = np.degrees(angle_rad)
                            # Non-orthogonality is deviation from 0 degrees
                            non_orthogonality = np.abs(angle_deg - 0.0)
                            if non_orthogonality > max_non_orthogonality:
                                max_non_orthogonality = non_orthogonality
                self.cell_non_orthogonality_values[ci] = max_non_orthogonality

        # 3. Connectivity Check
        # Check for unreferenced nodes (nodes not part of any cell)
        referenced_nodes = set()
        for conn in self.cell_connectivity:
            for node_idx in conn:
                referenced_nodes.add(node_idx)
        if len(referenced_nodes) != self.num_nodes:
            unreferenced_node_count = self.num_nodes - len(referenced_nodes)
            self.connectivity_issues.append(
                f"Found {unreferenced_node_count} unreferenced nodes."
            )

        # Check for duplicate cells (cells with identical connectivity)
        unique_cells = set()
        for i, conn in enumerate(self.cell_connectivity):
            sorted_conn = tuple(sorted(conn))  # Sort to handle permutations
            if sorted_conn in unique_cells:
                self.connectivity_issues.append(
                    f"Found duplicate cell at index {i} (connectivity: {conn})."
                )
            else:
                unique_cells.add(sorted_conn)

        # Check for non-manifold edges/faces (more than 2 cells sharing an edge/face)
        # This is partially covered by cell_neighbors, where a face should have 1 or 2 neighbors.
        # If a face_map entry has more than 2 elements, it's non-manifold.
        face_map: Dict[Tuple[int, ...], List[int]] = {}
        for ci, faces in enumerate(self.cell_faces):
            for face in faces:
                key = tuple(sorted(face))  # Use sorted tuple of node indices as key
                face_map.setdefault(key, []).append(ci)

        for face_key, cells_sharing_face in face_map.items():
            if len(cells_sharing_face) > 2:
                self.connectivity_issues.append(
                    f"Non-manifold face/edge detected (shared by {len(cells_sharing_face)} cells): {face_key}."
                )

    def print_summary(self) -> None:
        """Prints a nicely formatted summary of the mesh and its quality metrics."""
        if not self._is_analyzed:
            print("Mesh has not been analyzed. Please run analyze_mesh() first.")
            return

        print("\n" + "=" * 80)
        print(f"{'Mesh Analysis Report':^80}")
        print("=" * 80)

        # --- General Information ---
        print(f"\n{'--- General Information ---':^80}\n")
        print(f"  {'Dimension:' :<25} {self.dimension}D")
        print(f"  {'Number of Nodes:' :<25} {self.num_nodes}")
        print(f"  {'Number of Cells:' :<25} {self.num_cells}")

        if self.cell_type_map:
            print(f"  {'Cell Types:' :<25}")
            for type_id, props in self.cell_type_map.items():
                count = np.sum(self.cell_type_ids == type_id)
                print(f"    - {props['name']:<21} {count} cells")

        # --- Cell Geometry ---
        if self.cell_volumes.size > 0:
            print(f"\n{'--- Cell Geometry ---':^80}\n")
            print(f"  {'Metric':<20} {'Min':>15} {'Max':>15} {'Average':>15}")
            print(f"  {'-'*19} {'-'*15} {'-'*15} {'-'*15}")
            vol_min = np.min(self.cell_volumes)
            vol_max = np.max(self.cell_volumes)
            vol_avg = np.mean(self.cell_volumes)
            print(
                f"  {'Cell Volume':<20} {vol_min:>15.4e} {vol_max:>15.4e} {vol_avg:>15.4e}"
            )

        # --- Mesh Quality Metrics ---
        print(f"\n{'--- Mesh Quality Metrics ---':^80}\n")
        print(f"  {'Metric':<25} {'Min':>15} {'Max':>15} {'Average':>15}")
        print(f"  {'-'*24} {'-'*15} {'-'*15} {'-'*15}")

        if hasattr(self, "min_max_volume_ratio") and self.min_max_volume_ratio > 0:
            print(
                f"  {'Min/Max Volume Ratio':<25} {self.min_max_volume_ratio:>15.4f} {'-':>15} {'-':>15}"
            )

        if self.cell_skewness_values.size > 0:
            skew_min = np.min(self.cell_skewness_values)
            skew_max = np.max(self.cell_skewness_values)
            skew_avg = np.mean(self.cell_skewness_values)
            print(
                f"  {'Skewness':<25} {skew_min:>15.4f} {skew_max:>15.4f} {skew_avg:>15.4f}"
            )

        if self.cell_non_orthogonality_values.size > 0:
            non_ortho_min = np.min(self.cell_non_orthogonality_values)
            non_ortho_max = np.max(self.cell_non_orthogonality_values)
            non_ortho_avg = np.mean(self.cell_non_orthogonality_values)
            print(
                f"  {'Non-Orthogonality (deg)':<25} {non_ortho_min:>15.4f} {non_ortho_max:>15.4f} {non_ortho_avg:>15.4f}"
            )

        if self.cell_aspect_ratio_values.size > 0:
            ar_min = np.min(self.cell_aspect_ratio_values)
            ar_max = np.max(self.cell_aspect_ratio_values)
            ar_avg = np.mean(self.cell_aspect_ratio_values)
            print(
                f"  {'Aspect Ratio':<25} {ar_min:>15.4f} {ar_max:>15.4f} {ar_avg:>15.4f}"
            )

        # --- Connectivity Check ---
        print(f"\n{'--- Connectivity Check ---':^80}\n")
        if self.connectivity_issues:
            print("  Issues Found:")
            for issue in self.connectivity_issues:
                print(f"    - {issue}")
        else:
            print("  No connectivity issues found.")

        print("\n" + "=" * 80)


if __name__ == "__main__":
    print("mesh.py module. Import classes and use in your script.")
