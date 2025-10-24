# -*- coding: utf-8 -*-
"""
Represents a polygonal mesh with detailed geometric and quality information.

This module defines the `PolyMesh` class, which extends `CoreMesh` to include
derived geometric properties essential for finite volume methods. It provides a
comprehensive analysis of the mesh, making it suitable for numerical simulations.

Key Features:
- Computation of cell volumes, face areas, and face normals.
- Detailed geometric analysis of the mesh.
- Integration with mesh quality assessment tools.

Classes:
    PolyMesh: Extends CoreMesh with advanced geometric computations.
"""

from typing import List

import numpy as np

from .core_mesh import CoreMesh
from .quality import MeshQuality


class PolyMesh(CoreMesh):
    """
    Represents a polygonal mesh with derived geometric and quality information.

    This class extends CoreMesh to include detailed geometric properties
    essential for finite volume methods, such as cell volumes, face areas,
    and face normals. It provides a comprehensive analysis of the mesh,
    making it suitable for numerical simulations.

    Attributes:
        cell_volumes (np.ndarray): Volume of each cell.
        face_midpoints (np.ndarray): Midpoint of each face of each cell.
        face_normals (np.ndarray): Normal vector of each face of each cell.
        face_areas (np.ndarray): Area of each face of each cell.
        quality (MeshQuality): An object to compute and store mesh quality metrics.
    """

    def __init__(self) -> None:
        """Initializes the PolyMesh instance and its attributes."""
        super().__init__()

        # Computed geometric fields
        self.cell_volumes: np.ndarray = np.array([])
        self.face_midpoints: np.ndarray = np.array([])
        self.face_normals: np.ndarray = np.array([])
        self.face_areas: np.ndarray = np.array([])

        self.quality = MeshQuality()

    @classmethod
    def from_gmsh(cls, msh_file: str, gmsh_verbose: int = 0) -> "PolyMesh":
        """
        Creates a PolyMesh instance from a Gmsh .msh file.

        Args:
            msh_file (str): The path to the .msh file.
            gmsh_verbose (int): The verbosity level for the Gmsh API.

        Returns:
            A new PolyMesh instance populated with data from the file.
        """
        mesh = cls()
        if msh_file:
            mesh.read_gmsh(msh_file, gmsh_verbose)
        return mesh

    def analyze_mesh(self) -> None:
        """
        Computes all derived geometric and connectivity information for the mesh.
        """
        if self.num_cells == 0:
            raise RuntimeError("No cells found. Read a mesh first.")
        super().analyze_mesh()
        self._compute_face_properties()
        self._compute_cell_volumes()
        self._is_analyzed = True

    def _compute_face_properties(self) -> None:
        """Computes midpoint, area, and normal for each face of each cell."""
        if self.cell_neighbors.size == 0:
            return

        max_faces = self.cell_neighbors.shape[1]
        self.face_midpoints = np.zeros((self.num_cells, max_faces, 3))
        self.face_normals = np.zeros((self.num_cells, max_faces, 3))
        self.face_areas = np.zeros((self.num_cells, max_faces))

        for ci, faces in enumerate(self.cell_faces):
            for fi, face_nodes in enumerate(faces):
                nodes = self.node_coords[face_nodes]
                self.face_midpoints[ci, fi] = np.mean(nodes, axis=0)
                if self.dimension == 2:
                    self._compute_2d_face_metrics(ci, fi, nodes)
                elif self.dimension == 3:
                    self._compute_3d_face_metrics(ci, fi, nodes)

        self._orient_face_normals()

    def _compute_2d_face_metrics(self, ci: int, fi: int, nodes: np.ndarray) -> None:
        """Computes area and normal for a 2D face (an edge)."""
        edge = nodes[1] - nodes[0]
        length = np.linalg.norm(edge)
        self.face_areas[ci, fi] = length
        if length > 1e-9:
            self.face_normals[ci, fi] = np.array([edge[1], -edge[0], 0.0]) / length

    def _compute_3d_face_metrics(self, ci: int, fi: int, nodes: np.ndarray) -> None:
        """Computes area and normal for a 3D face (a polygon)."""
        centroid = np.mean(nodes, axis=0)
        area_vec = (
            sum(
                np.cross(nodes[k] - centroid, nodes[(k + 1) % len(nodes)] - centroid)
                for k in range(len(nodes))
            )
            * 0.5
        )
        area = np.linalg.norm(area_vec)
        self.face_areas[ci, fi] = area
        if area > 1e-9:
            self.face_normals[ci, fi] = area_vec / area

    def _orient_face_normals(self) -> None:
        """Ensures all face normals point outwards from the cell centroid."""
        for ci in range(self.num_cells):
            for fi in range(len(self.cell_faces[ci])):
                vec_to_face = self.face_midpoints[ci, fi] - self.cell_centroids[ci]
                if np.dot(self.face_normals[ci, fi], vec_to_face) < 0:
                    self.face_normals[ci, fi] *= -1

    def _compute_cell_volumes(self) -> None:
        """Computes the volume of each cell."""
        if self.dimension == 2:
            self._compute_2d_cell_volumes()
        elif self.dimension == 3:
            self._compute_3d_cell_volumes()
        else:
            self.cell_volumes = np.zeros(self.num_cells)

    def _compute_2d_cell_volumes(self) -> None:
        """Computes cell areas for a 2D mesh using the shoelace formula."""
        volumes = np.zeros(self.num_cells)
        for i, conn in enumerate(self.cell_connectivity):
            nodes = self.node_coords[conn][:, :2]
            x, y = nodes[:, 0], nodes[:, 1]
            volumes[i] = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        self.cell_volumes = volumes

    def _compute_3d_cell_volumes(self) -> None:
        """Computes cell volumes for a 3D mesh using the divergence theorem."""
        contrib = (self.face_midpoints * self.face_normals).sum(
            axis=2
        ) * self.face_areas
        self.cell_volumes = np.sum(contrib, axis=1) / 3.0

    def print_summary(self) -> None:
        """Prints a formatted summary of the mesh and its quality metrics."""
        if not self._is_analyzed:
            print("Mesh not analyzed. Run analyze_mesh() first.")
            return

        print("\n" + "=" * 80)
        print(f"{"Mesh Analysis Report":^80}")
        print("=" * 80)
        self._print_general_info()
        self._print_geometric_properties()
        self._print_cell_geometry()
        self._compute_quality()
        self.quality.print_summary()
        print("\n" + "=" * 80)

    def _print_general_info(self) -> None:
        """Prints general information about the mesh."""
        print(f"\n{'--- General Information ---':^80}\n")
        print(f"  {'Dimension:':<25} {self.dimension}D")
        print(f"  {'Number of Nodes:':<25} {self.num_nodes}")
        print(f"  {'Number of Cells:':<25} {self.num_cells}")

    def _print_geometric_properties(self) -> None:
        """Prints the geometric bounds of the mesh."""
        if self.num_nodes == 0:
            return
        x_min, y_min, z_min = np.min(self.node_coords, axis=0)
        x_max, y_max, z_max = np.max(self.node_coords, axis=0)
        print(f"\n{'--- Geometric Properties ---':^80}\n")
        print(f"  {'X Range:':<25} {x_min:.4f} to {x_max:.4f}")
        print(f"  {'Y Range:':<25} {y_min:.4f} to {y_max:.4f}")
        if self.dimension == 3:
            print(f"  {'Z Range:':<25} {z_min:.4f} to {z_max:.4f}")

    def _print_cell_geometry(self) -> None:
        """Prints information about cell geometry and types."""
        if self.cell_volumes.size == 0:
            return
        print(f"\n{'--- Cell Geometry ---':^80}\n")
        self._print_cell_type_distribution()
        self._print_cell_volume_stats()

    def _print_cell_type_distribution(self) -> None:
        """Prints the distribution of different cell types."""
        cell_type_counts = {}
        for conn in self.cell_connectivity:
            if len(conn) > 4:
                label = f"{len(conn)}-gon"
            else:
                label = {3: "Triangle", 4: "Quadrilateral"}.get(
                    len(conn), f"{len(conn)}-gon"
                )
            cell_type_counts[label] = cell_type_counts.get(label, 0) + 1
        print("  Cell Type Distribution:")
        for label, count in cell_type_counts.items():
            print(f"    - {label+':':<20} {count}")

    def _print_cell_volume_stats(self) -> None:
        """Prints statistics about cell volumes."""
        print(f"\n  {'Metric':<20} {'Min':>15} {'Max':>15} {'Average':>15}")
        print(f"  {'-'*19} {'-'*15} {'-'*15} {'-'*15}")
        vol_min, vol_max, vol_avg = (
            np.min(self.cell_volumes),
            np.max(self.cell_volumes),
            np.mean(self.cell_volumes),
        )
        print(
            f"  {'Cell Volume':<20} {vol_min:>15.4e} {vol_max:>15.4e} {vol_avg:>15.4e}"
        )

    def _compute_quality(self) -> None:
        """Computes all available mesh quality metrics."""
        self.quality.compute(self)
