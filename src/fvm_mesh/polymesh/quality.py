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
from __future__ import annotations
from typing import Dict, List, Tuple, TYPE_CHECKING
from dataclasses import dataclass
import numpy as np

if TYPE_CHECKING:
    from .poly_mesh import PolyMesh

# --- Constants for magic numbers ---
GEOMETRY_TOLERANCE = 1e-12
IDEAL_TRIANGLE_ANGLE = 60.0
IDEAL_QUAD_ANGLE = 90.0


@dataclass(frozen=True)
class MeshQuality:
    """
    Stores mesh quality metrics for a PolyMesh object.

    This is a data-centric class that holds the results of a quality analysis.
    Instances of this class are created via the `from_mesh` class method.

    Attributes:
        min_max_volume_ratio (float): Ratio of the smallest to the largest cell volume.
        cell_skewness_values (np.ndarray): Skewness values for each cell.
        cell_non_orthogonality_values (np.ndarray): Non-orthogonality values (in degrees)
            for each cell.
        cell_aspect_ratio_values (np.ndarray): Aspect ratio values for each cell.
        connectivity_issues (List[str]): A list of strings describing any
            topological issues found in the mesh.
    """

    min_max_volume_ratio: float
    cell_skewness_values: np.ndarray
    cell_non_orthogonality_values: np.ndarray
    cell_aspect_ratio_values: np.ndarray
    connectivity_issues: List[str]

    @classmethod
    def from_mesh(cls, mesh: "PolyMesh") -> "MeshQuality":
        """
        Computes all mesh quality metrics from a PolyMesh object and returns a new instance.
        """
        if not mesh._is_analyzed:
            raise RuntimeError("Mesh must be analyzed before computing quality.")

        if mesh.n_cells == 0:
            return cls(
                min_max_volume_ratio=0.0,
                cell_skewness_values=np.array([]),
                cell_non_orthogonality_values=np.array([]),
                cell_aspect_ratio_values=np.array([]),
                connectivity_issues=[],
            )

        min_max_volume_ratio = cls._compute_volume_ratio(mesh)
        skewness, aspect_ratio = cls._compute_geometric_metrics(mesh)
        non_orthogonality = cls._compute_non_orthogonality(mesh)
        connectivity_issues = cls._check_connectivity(mesh)

        return cls(
            min_max_volume_ratio=min_max_volume_ratio,
            cell_skewness_values=skewness,
            cell_non_orthogonality_values=non_orthogonality,
            cell_aspect_ratio_values=aspect_ratio,
            connectivity_issues=connectivity_issues,
        )

    @staticmethod
    def _compute_volume_ratio(mesh: "PolyMesh") -> float:
        """Calculates the ratio of the smallest to the largest cell volume."""
        min_vol = np.min(mesh.cell_volumes)
        max_vol = np.max(mesh.cell_volumes)
        return min_vol / max_vol if max_vol > GEOMETRY_TOLERANCE else 0.0

    @staticmethod
    def _compute_geometric_metrics(mesh: "PolyMesh") -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes cell-based geometric metrics like skewness and aspect ratio.
        """
        n_cells = mesh.n_cells
        skewness = np.zeros(n_cells)
        aspect_ratio = np.zeros(n_cells)

        # Note: This implementation is limited to 2D meshes with triangle and quad cells.
        # To expand, add handlers for 3D cell types and other polygon shapes.
        if mesh.dimension == 2:
            for i, conn in enumerate(mesh.cell_node_connectivity):
                nodes = mesh.node_coords[conn][:, :2]
                if len(nodes) == 3:
                    s, ar = MeshQuality._compute_triangle_metrics(nodes)
                    skewness[i] = s
                    aspect_ratio[i] = ar
                elif len(nodes) == 4:
                    s, ar = MeshQuality._compute_quad_metrics(nodes)
                    skewness[i] = s
                    aspect_ratio[i] = ar

        return skewness, aspect_ratio

    @staticmethod
    def _compute_triangle_metrics(nodes: np.ndarray) -> Tuple[float, float]:
        """Computes skewness and aspect ratio for a single triangular cell."""
        lengths = np.linalg.norm(np.roll(nodes, -1, axis=0) - nodes, axis=1)
        if np.min(lengths) < GEOMETRY_TOLERANCE:
            return 1.0, np.inf

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
        skewness = np.max(np.abs(angles - IDEAL_TRIANGLE_ANGLE)) / IDEAL_TRIANGLE_ANGLE

        # Aspect ratio
        aspect_ratio = np.max(lengths) / np.min(lengths)
        return skewness, aspect_ratio

    @staticmethod
    def _compute_quad_metrics(nodes: np.ndarray) -> Tuple[float, float]:
        """Computes skewness and aspect ratio for a single quadrilateral cell."""
        edge_lengths = np.linalg.norm(np.roll(nodes, -1, axis=0) - nodes, axis=1)
        if np.min(edge_lengths) < GEOMETRY_TOLERANCE:
            return 1.0, np.inf

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
        skewness = np.max(np.abs(angles - IDEAL_QUAD_ANGLE)) / IDEAL_QUAD_ANGLE

        # Aspect ratio
        aspect_ratio = np.max(edge_lengths) / np.min(edge_lengths)
        return skewness, aspect_ratio

    @staticmethod
    def _compute_non_orthogonality(mesh: "PolyMesh") -> np.ndarray:
        """
        Computes the maximum non-orthogonality for each cell.
        """
        non_orthogonality = np.zeros(mesh.n_cells)
        for ci in range(mesh.n_cells):
            max_non_ortho = 0.0
            for fi, _ in enumerate(mesh.cell_face_nodes[ci]):
                if fi >= mesh.cell_face_midpoints.shape[1]:
                    continue
                vec_to_face = mesh.cell_face_midpoints[ci, fi] - mesh.cell_centroids[ci]
                norm_vec = np.linalg.norm(vec_to_face)
                norm_normal = np.linalg.norm(mesh.cell_face_normals[ci, fi])

                if norm_vec > GEOMETRY_TOLERANCE and norm_normal > GEOMETRY_TOLERANCE:
                    dot_p = np.dot(vec_to_face, mesh.cell_face_normals[ci, fi])
                    cos_angle = np.clip(dot_p / (norm_vec * norm_normal), -1.0, 1.0)
                    angle_deg = np.degrees(np.arccos(cos_angle))
                    max_non_ortho = max(max_non_ortho, angle_deg)
            non_orthogonality[ci] = max_non_ortho
        return non_orthogonality

    @staticmethod
    def _check_connectivity(mesh: "PolyMesh") -> List[str]:
        """Checks for topological issues like unreferenced nodes."""
        issues = []
        referenced_nodes = set(
            node for conn in mesh.cell_node_connectivity for node in conn
        )
        if len(referenced_nodes) < mesh.n_nodes:
            unreferenced = set(range(mesh.n_nodes)) - referenced_nodes
            issues.append(f"Found {len(unreferenced)} unreferenced nodes.")

        # Check for duplicate cells
        unique_cells = {tuple(sorted(conn)) for conn in mesh.cell_node_connectivity}
        if len(unique_cells) < mesh.n_cells:
            issues.append("Found duplicate cells.")

        face_map: Dict[Tuple[int, ...], List[int]] = {}
        for ci, faces in enumerate(mesh.cell_face_nodes):
            for face in faces:
                key = tuple(sorted(face))
                face_map.setdefault(key, []).append(ci)
        for key, cells in face_map.items():
            if len(cells) > 2:
                issues.append(f"Non-manifold face {key} shared by {len(cells)} cells.")
        return issues
