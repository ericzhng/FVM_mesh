# -*- coding: utf-8 -*-
"""
Computes and stores mesh quality metrics for a PolyMesh object.

This module provides the `MeshQuality` class, which calculates various metrics
to assess the quality of a mesh. These metrics are crucial for ensuring the
stability and accuracy of numerical simulations.

Key Features:
- Calculation of geometric metrics like volume ratio, skewness, and aspect ratio.
- Computation of non-orthogonality to check face alignment.
- Topological checks for issues like unreferenced nodes and non-manifold faces.

Classes:
    MeshQuality: A class for computing and storing mesh quality metrics.
"""

import numpy as np
from typing import Dict, List, Tuple


class MeshQuality:
    """
    Computes and stores mesh quality metrics for a PolyMesh object.

    This class calculates various metrics to assess the quality of a mesh,
    which is crucial for the stability and accuracy of numerical simulations.

    Attributes:
        min_max_volume_ratio (float): Ratio of the smallest to the largest cell volume.
        cell_skewness_values (np.ndarray): Skewness values for each cell.
        cell_non_orthogonality_values (np.ndarray): Non-orthogonality values (in degrees)
            for each cell.
        cell_aspect_ratio_values (np.ndarray): Aspect ratio values for each cell.
        connectivity_issues (List[str]): A list of strings describing any
            topological issues found in the mesh.
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
        Computes all mesh quality metrics from a PolyMesh object.

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
        """Resets all metric attributes to their initial state."""
        self.min_max_volume_ratio = 0.0
        self.cell_skewness_values = np.zeros(num_cells)
        self.cell_non_orthogonality_values = np.zeros(num_cells)
        self.cell_aspect_ratio_values = np.zeros(num_cells)
        self.connectivity_issues = []
        self._is_computed = False

    def _compute_volume_ratio(self, mesh) -> None:
        """Calculates the ratio of the smallest to the largest cell volume."""
        min_vol = np.min(mesh.cell_volumes)
        max_vol = np.max(mesh.cell_volumes)
        self.min_max_volume_ratio = min_vol / max_vol if max_vol > 1e-12 else 0.0

    def _compute_geometric_metrics(self, mesh) -> None:
        """
        Computes cell-based geometric metrics like skewness and aspect ratio.
        """
        if mesh.dimension != 2:
            return  # Implemented only for 2D meshes currently.

        for i, conn in enumerate(mesh.cell_connectivity):
            nodes = mesh.node_coords[conn][:, :2]
            if len(nodes) == 3:
                self._compute_triangle_metrics(i, nodes)
            elif len(nodes) == 4:
                self._compute_quad_metrics(i, nodes)

        self._compute_non_orthogonality(mesh)

    def _compute_triangle_metrics(self, index: int, nodes: np.ndarray) -> None:
        """Computes skewness and aspect ratio for a single triangular cell."""
        lengths = np.linalg.norm(np.roll(nodes, -1, axis=0) - nodes, axis=1)
        if np.min(lengths) < 1e-12:
            self.cell_skewness_values[index] = 1.0
            self.cell_aspect_ratio_values[index] = np.inf
            return

        # Angle-based skewness
        angles = np.degrees(
            [
                np.arccos(
                    np.clip(
                        np.dot(
                            nodes[(i + 1) % 3] - nodes[i],
                            nodes[(i - 1 + 3) % 3] - nodes[i],
                        )
                        / (lengths[i] * lengths[(i - 1 + 3) % 3]),
                        -1.0,
                        1.0,
                    )
                )
                for i in range(3)
            ]
        )
        self.cell_skewness_values[index] = np.max(np.abs(angles - 60.0)) / 60.0

        # Aspect ratio
        self.cell_aspect_ratio_values[index] = np.max(lengths) / np.min(lengths)

    def _compute_quad_metrics(self, index: int, nodes: np.ndarray) -> None:
        """Computes skewness and aspect ratio for a single quadrilateral cell."""
        # Edge lengths
        edge_lengths = np.linalg.norm(np.roll(nodes, -1, axis=0) - nodes, axis=1)
        if np.min(edge_lengths) < 1e-12:
            self.cell_skewness_values[index] = 1.0
            self.cell_aspect_ratio_values[index] = np.inf
            return

        # Angle-based skewness
        angles = np.degrees(
            [
                np.arccos(
                    np.clip(
                        np.dot(
                            nodes[(j - 1 + 4) % 4] - nodes[j],
                            nodes[(j + 1) % 4] - nodes[j],
                        )
                        / (
                            np.linalg.norm(nodes[(j - 1 + 4) % 4] - nodes[j])
                            * np.linalg.norm(nodes[(j + 1) % 4] - nodes[j])
                        ),
                        -1.0,
                        1.0,
                    )
                )
                for j in range(4)
            ]
        )
        self.cell_skewness_values[index] = np.max(np.abs(angles - 90.0)) / 90.0

        # Aspect ratio
        self.cell_aspect_ratio_values[index] = np.max(edge_lengths) / np.min(
            edge_lengths
        )

    def _compute_non_orthogonality(self, mesh) -> None:
        """
        Computes the maximum non-orthogonality for each cell.
        """
        for ci in range(mesh.num_cells):
            max_non_ortho = 0.0
            for fi, _ in enumerate(mesh.cell_faces[ci]):
                if fi >= mesh.face_midpoints.shape[1]:
                    continue
                vec_to_face = mesh.face_midpoints[ci, fi] - mesh.cell_centroids[ci]
                norm_vec = np.linalg.norm(vec_to_face)
                norm_normal = np.linalg.norm(mesh.face_normals[ci, fi])

                if norm_vec > 1e-12 and norm_normal > 1e-12:
                    dot_p = np.dot(vec_to_face, mesh.face_normals[ci, fi])
                    cos_angle = np.clip(dot_p / (norm_vec * norm_normal), -1.0, 1.0)
                    angle_deg = np.degrees(np.arccos(cos_angle))
                    max_non_ortho = max(max_non_ortho, angle_deg)
            self.cell_non_orthogonality_values[ci] = max_non_ortho

    def _check_connectivity(self, mesh) -> None:
        """Checks for topological issues like unreferenced nodes."""
        referenced_nodes = set(node for conn in mesh.cell_connectivity for node in conn)
        if len(referenced_nodes) < mesh.num_nodes:
            unreferenced = set(range(mesh.num_nodes)) - referenced_nodes
            self.connectivity_issues.append(
                f"Found {len(unreferenced)} unreferenced nodes."
            )

        # Check for duplicate cells
        unique_cells = {tuple(sorted(conn)) for conn in mesh.cell_connectivity}
        if len(unique_cells) < mesh.num_cells:
            self.connectivity_issues.append("Found duplicate cells.")

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
        Prints a formatted summary of the computed mesh quality metrics.
        """
        if not self._is_computed:
            print("Quality metrics not computed. Run compute() first.")
            return

        print(f"\n{'--- Mesh Quality Metrics ---':^80}\n")
        self._print_metric_table()
        self._print_connectivity_issues()

    def _print_metric_table(self) -> None:
        """Prints the table of quality metrics."""
        print(f"  {'Metric':<25} {'Min':>15} {'Max':>15} {'Average':>15}")
        print(f"  {'-'*24} {'-'*15} {'-'*15} {'-'*15}")

        self._print_metric_row(
            "Min/Max Volume Ratio",
            np.array(self.min_max_volume_ratio),
            is_single_value=True,
        )
        self._print_metric_row("Skewness", self.cell_skewness_values)
        self._print_metric_row(
            "Non-Orthogonality (deg)", self.cell_non_orthogonality_values
        )
        self._print_metric_row(
            "Aspect Ratio", self.cell_aspect_ratio_values, filter_finite=True
        )

    def _print_metric_row(
        self,
        name: str,
        values: np.ndarray,
        is_single_value: bool = False,
        filter_finite: bool = False,
    ) -> None:
        """Prints a single row in the metric table."""
        if is_single_value:
            if values > 0:
                print(f"  {name:<25} {values:>15.4f} {'-':>15} {'-':>15}")
            return

        if values.size > 0 and np.any(values):
            if filter_finite:
                values = values[np.isfinite(values)]
            if values.size > 0:
                min_val, max_val, mean_val = (
                    np.min(values),
                    np.max(values),
                    np.mean(values),
                )
                print(
                    f"  {name:<25} {min_val:>15.4f} {max_val:>15.4f} {mean_val:>15.4f}"
                )

    def _print_connectivity_issues(self) -> None:
        """Prints any connectivity issues found."""
        print(f"\n{'--- Connectivity Check ---':^80}\n")
        if self.connectivity_issues:
            print("  Issues Found:")
            for issue in self.connectivity_issues:
                print(f"    - {issue}")
        else:
            print("  No connectivity issues found.")
