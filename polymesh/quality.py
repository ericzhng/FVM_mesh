# -*- coding: utf-8 -*-
import numpy as np
from typing import Dict, List, Tuple


class MeshQuality:
    """
    Computes and stores mesh quality metrics for a PolyMesh object.

    This class calculates various metrics to assess the quality of a mesh,
    which is crucial for the stability and accuracy of numerical simulations.
    Metrics include geometric properties like volume ratios, skewness, and
    aspect ratio, as well as topological checks for connectivity issues.

    Attributes:
        min_max_volume_ratio (float): Ratio of the smallest cell volume to the
            largest cell volume in the mesh.
        cell_skewness_values (np.ndarray): An array of skewness values for each cell.
        cell_non_orthogonality_values (np.ndarray): An array of non-orthogonality
            values (in degrees) for each cell.
        cell_aspect_ratio_values (np.ndarray): An array of aspect ratio values for
            each cell.
        connectivity_issues (List[str]): A list of strings describing any
            topological issues found in the mesh (e.g., unreferenced nodes).
    """

    def __init__(self):
        """Initializes the MeshQuality instance with default values."""
        self.min_max_volume_ratio: float = 0.0
        self.cell_skewness_values: np.ndarray = np.array([])
        self.cell_non_orthogonality_values: np.ndarray = np.array([])
        self.cell_aspect_ratio_values: np.ndarray = np.array([])
        self.connectivity_issues: List[str] = []
        self._is_computed = False

    def compute(self, mesh) -> None:
        """
        Compute all mesh quality metrics from a PolyMesh object.

        Args:
            mesh (PolyMesh): The mesh object to be analyzed.
        """
        if not mesh._is_analyzed:
            raise RuntimeError("Mesh must be analyzed before computing quality.")

        self._initialize_metrics(mesh.num_cells)

        if mesh.num_cells == 0:
            return

        self._compute_volume_ratio(mesh)
        self._compute_geometric_metrics(mesh)
        self._check_connectivity(mesh)

        self._is_computed = True

    def _initialize_metrics(self, num_cells: int) -> None:
        """Reset all metric attributes to their initial state."""
        self.min_max_volume_ratio = 0.0
        self.cell_skewness_values = np.zeros(num_cells)
        self.cell_non_orthogonality_values = np.zeros(num_cells)
        self.cell_aspect_ratio_values = np.zeros(num_cells)
        self.connectivity_issues = []
        self._is_computed = False

    def _compute_volume_ratio(self, mesh) -> None:
        """Calculate the ratio of the smallest to the largest cell volume."""
        min_volume = np.min(mesh.cell_volumes)
        max_volume = np.max(mesh.cell_volumes)
        self.min_max_volume_ratio = min_volume / max_volume if max_volume > 1e-12 else 0.0

    def _compute_geometric_metrics(self, mesh) -> None:
        """
        Compute cell-based geometric metrics like skewness, aspect ratio,
        and non-orthogonality.
        """
        if mesh.dimension != 2:
            # Currently, these metrics are implemented only for 2D meshes.
            return

        for i, conn in enumerate(mesh.cell_connectivity):
            nodes = mesh.node_coords[conn][:, :2]
            num_nodes = len(nodes)

            if num_nodes == 3:  # Triangle
                self._compute_triangle_metrics(i, nodes)
            elif num_nodes == 4:  # Quadrilateral
                self._compute_quad_metrics(i, nodes)

        self._compute_non_orthogonality(mesh)

    def _compute_triangle_metrics(self, index: int, nodes: np.ndarray) -> None:
        """Compute skewness and aspect ratio for a single triangular cell."""
        # Edge vectors and lengths
        v0 = nodes[1] - nodes[0]
        v1 = nodes[2] - nodes[1]
        v2 = nodes[0] - nodes[2]
        lengths = np.array([np.linalg.norm(v) for v in [v0, v1, v2]])

        if np.min(lengths) < 1e-12:
            self.cell_skewness_values[index] = 1.0  # Degenerate
            self.cell_aspect_ratio_values[index] = np.inf
            return

        # Angle-based skewness
        angles = np.degrees([
            np.arccos(np.dot(-v2, v0) / (lengths[2] * lengths[0])),
            np.arccos(np.dot(-v0, v1) / (lengths[0] * lengths[1])),
            np.arccos(np.dot(-v1, v2) / (lengths[1] * lengths[2]))
        ])
        self.cell_skewness_values[index] = np.max(np.abs(angles - 60.0)) / 60.0

        # Aspect ratio
        self.cell_aspect_ratio_values[index] = np.max(lengths) / np.min(lengths)

    def _compute_quad_metrics(self, index: int, nodes: np.ndarray) -> None:
        """Compute skewness and aspect ratio for a single quadrilateral cell."""
        # Edge lengths
        edge_lengths = np.array([
            np.linalg.norm(nodes[j] - nodes[(j + 1) % 4]) for j in range(4)
        ])

        if np.min(edge_lengths) < 1e-12:
            self.cell_skewness_values[index] = 1.0  # Degenerate
            self.cell_aspect_ratio_values[index] = np.inf
            return

        # Angle-based skewness
        angles = []
        for j in range(4):
            v1 = nodes[(j - 1 + 4) % 4] - nodes[j]
            v2 = nodes[(j + 1) % 4] - nodes[j]
            dot_p = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angles.append(np.degrees(np.arccos(np.clip(dot_p, -1.0, 1.0))))
        self.cell_skewness_values[index] = np.max(np.abs(np.array(angles) - 90.0)) / 90.0

        # Aspect ratio
        self.cell_aspect_ratio_values[index] = np.max(edge_lengths) / np.min(edge_lengths)

    def _compute_non_orthogonality(self, mesh) -> None:
        """
        Compute the maximum non-orthogonality for each cell.

        Non-orthogonality is the angle between the vector connecting a cell
        centroid to a face midpoint and the face normal vector.
        """
        for ci in range(mesh.num_cells):
            max_non_ortho = 0.0
            for fi, _ in enumerate(mesh.cell_faces[ci]):
                if fi >= mesh.face_midpoints.shape[1]: continue

                vec_to_face = mesh.face_midpoints[ci, fi] - mesh.cell_centroids[ci]
                norm_vec = np.linalg.norm(vec_to_face)
                norm_normal = np.linalg.norm(mesh.face_normals[ci, fi])

                if norm_vec > 1e-12 and norm_normal > 1e-12:
                    dot_p = np.dot(vec_to_face, mesh.face_normals[ci, fi])
                    cos_angle = dot_p / (norm_vec * norm_normal)
                    angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
                    max_non_ortho = max(max_non_ortho, angle_deg)

            self.cell_non_orthogonality_values[ci] = max_non_ortho

    def _check_connectivity(self, mesh) -> None:
        """Check for topological issues like unreferenced nodes and non-manifold faces."""
        # Check for unreferenced nodes
        all_nodes = set(range(mesh.num_nodes))
        referenced_nodes = set(node for conn in mesh.cell_connectivity for node in conn)
        unreferenced = all_nodes - referenced_nodes
        if unreferenced:
            self.connectivity_issues.append(f"Found {len(unreferenced)} unreferenced nodes.")

        # Check for duplicate cells
        unique_cells = set()
        for i, conn in enumerate(mesh.cell_connectivity):
            sorted_conn = tuple(sorted(conn))
            if sorted_conn in unique_cells:
                self.connectivity_issues.append(f"Duplicate cell at index {i}: {conn}")
            unique_cells.add(sorted_conn)

        # Check for non-manifold faces (more than 2 cells sharing a face)
        face_map: Dict[Tuple[int, ...], List[int]] = {}
        for ci, faces in enumerate(mesh.cell_faces):
            for face in faces:
                key = tuple(sorted(face))
                face_map.setdefault(key, []).append(ci)

        for key, cells in face_map.items():
            if len(cells) > 2:
                self.connectivity_issues.append(
                    f"Non-manifold face {key} shared by {len(cells)} cells."
                )

    def print_summary(self) -> None:
        """
        Prints a nicely formatted summary of the computed mesh quality metrics.
        """
        if not self._is_computed:
            print("Quality metrics have not been computed. Run compute() first.")
            return

        print(f"\n{'--- Mesh Quality Metrics ---':^80}\n")
        print(f"  {'Metric':<25} {'Min':>15} {'Max':>15} {'Average':>15}")
        print(f"  {'-'*24} {'-'*15} {'-'*15} {'-'*15}")

        if self.min_max_volume_ratio > 0:
            print(
                f"  {'Min/Max Volume Ratio':<25} {self.min_max_volume_ratio:>15.4f} {'-':>15} {'-':>15}"
            )

        if self.cell_skewness_values.size > 0 and np.any(self.cell_skewness_values):
            vals = self.cell_skewness_values
            print(f"  {'Skewness':<25} {np.min(vals):>15.4f} {np.max(vals):>15.4f} {np.mean(vals):>15.4f}")

        if self.cell_non_orthogonality_values.size > 0 and np.any(self.cell_non_orthogonality_values):
            vals = self.cell_non_orthogonality_values
            print(f"  {'Non-Orthogonality (deg)':<25} {np.min(vals):>15.4f} {np.max(vals):>15.4f} {np.mean(vals):>15.4f}")

        if self.cell_aspect_ratio_values.size > 0 and np.any(self.cell_aspect_ratio_values):
            vals = self.cell_aspect_ratio_values[np.isfinite(self.cell_aspect_ratio_values)]
            if vals.size > 0:
                print(f"  {'Aspect Ratio':<25} {np.min(vals):>15.4f} {np.max(vals):>15.4f} {np.mean(vals):>15.4f}")

        print(f"\n{'--- Connectivity Check ---':^80}\n")
        if self.connectivity_issues:
            print("  Issues Found:")
            for issue in self.connectivity_issues:
                print(f"    - {issue}")
        else:
            print("  No connectivity issues found.")
