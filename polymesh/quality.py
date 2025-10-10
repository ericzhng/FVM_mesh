import numpy as np
from typing import Dict, List, Tuple


class MeshQuality:
    """
    A data structure to store and compute mesh quality metrics.
    """

    def __init__(self):
        self.min_max_volume_ratio: float = 0.0
        self.cell_skewness_values: np.ndarray = np.array([])
        self.cell_non_orthogonality_values: np.ndarray = np.array([])
        self.cell_aspect_ratio_values: np.ndarray = np.array([])
        self.connectivity_issues: List[str] = []
        self._is_computed = False

    def compute(self, mesh):
        """
        Computes mesh quality metrics from a PolyMesh object.
        """
        if not mesh._is_analyzed:
            raise RuntimeError("Mesh must be analyzed before computing quality.")

        # Initialize quality metrics
        self.min_max_volume_ratio = 0.0
        self.cell_skewness_values = np.zeros(mesh.num_cells)
        self.cell_non_orthogonality_values = np.zeros(mesh.num_cells)
        self.cell_aspect_ratio_values = np.zeros(mesh.num_cells)
        self.connectivity_issues = []

        if mesh.num_cells == 0:
            return

        # 1. Ratio of smallest to biggest cell volume
        min_volume = np.min(mesh.cell_volumes)
        max_volume = np.max(mesh.cell_volumes)
        if max_volume > 0:
            self.min_max_volume_ratio = min_volume / max_volume
        else:
            self.min_max_volume_ratio = 0.0

        # Ensure cell_faces is populated for non-orthogonality and aspect ratio
        if not mesh.cell_faces:
            mesh._extract_cell_faces()  # Re-extract if not already done

        # 2. Skewness, Non-Orthogonality, and Aspect Ratio (2D only for now)
        if mesh.dimension == 2:
            for i, conn in enumerate(mesh.cell_connectivity):
                nodes = mesh.node_coords[conn][:, :2]
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
            for ci in range(mesh.num_cells):
                cell_centroid = mesh.cell_centroids[ci]
                max_non_orthogonality = 0.0
                for fi, face_nodes in enumerate(mesh.cell_faces[ci]):
                    if fi < len(mesh.face_midpoints[ci]) and fi < len(
                        mesh.face_normals[ci]
                    ):
                        face_midpoint = mesh.face_midpoints[ci, fi]
                        face_normal = mesh.face_normals[ci, fi]

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
        for conn in mesh.cell_connectivity:
            for node_idx in conn:
                referenced_nodes.add(node_idx)
        if len(referenced_nodes) != mesh.num_nodes:
            unreferenced_node_count = mesh.num_nodes - len(referenced_nodes)
            self.connectivity_issues.append(
                f"Found {unreferenced_node_count} unreferenced nodes."
            )

        # Check for duplicate cells (cells with identical connectivity)
        unique_cells = set()
        for i, conn in enumerate(mesh.cell_connectivity):
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
        for ci, faces in enumerate(mesh.cell_faces):
            for face in faces:
                key = tuple(sorted(face))  # Use sorted tuple of node indices as key
                face_map.setdefault(key, []).append(ci)

        for face_key, cells_sharing_face in face_map.items():
            if len(cells_sharing_face) > 2:
                self.connectivity_issues.append(
                    f"Non-manifold face/edge detected (shared by {len(cells_sharing_face)} cells): {face_key}."
                )
        
        self._is_computed = True

    def print_summary(self):
        """
        Prints a summary of the mesh quality metrics.
        """
        if not self._is_computed:
            print("Quality metrics have not been computed. Please run compute() first.")
            return

        print(f"\n{'--- Mesh Quality Metrics ---':^80}\n")
        print(f"  {'Metric':<25} {'Min':>15} {'Max':>15} {'Average':>15}")
        print(f"  {'-'*24} {'-'*15} {'-'*15} {'-'*15}")

        if self.min_max_volume_ratio > 0:
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
