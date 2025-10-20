# -*- coding: utf-8 -*-
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
        cell_centroids (np.ndarray): Centroids of each cell.
        cell_volumes (np.ndarray): Volume of each cell.
        cell_faces (List[List[List[int]]]): List of faces for each cell, where
            each face is a list of node indices.
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

        # Quality metrics
        self.quality = MeshQuality()

    @classmethod
    def from_gmsh(cls, msh_file: str, gmsh_verbose: int = 0) -> "PolyMesh":
        """
        Create a PolyMesh instance by reading a Gmsh .msh file.

        Args:
            msh_file (str): The path to the .msh file.
            gmsh_verbose (int): The verbosity level for the Gmsh API.

        Returns:
            PolyMesh: A new PolyMesh instance populated with data from the file.
        """
        mesh = cls()
        if msh_file:
            mesh.read_gmsh(msh_file, gmsh_verbose)
        return mesh

    def analyze_mesh(self) -> None:
        """
        Compute all derived geometric and connectivity information for the mesh.

        This method orchestrates the computation of centroids, neighbors, faces,
        face properties (areas, normals), and cell volumes. It should be called
        before attempting to use the mesh for simulations.
        """
        if self.num_cells == 0:
            raise RuntimeError(
                "No cells available. Call read_gmsh or populate cells first."
            )
        super().analyze_mesh()  # Computes centroids, neighbors, and cell_faces

        self._compute_face_midpoints_areas_normals()
        self._compute_cell_volumes()
        self._is_analyzed = True  # Set flag after successful analysis

    def _compute_face_midpoints_areas_normals(self) -> None:
        """
        Compute the midpoint, area, and normal vector for each face of each cell.

        The normal vectors are oriented to point outwards from the cell.
        For 2D cells, the "area" is the length of the edge.
        """
        if self.cell_neighbors.size == 0:
            self.face_midpoints = np.array([])
            self.face_normals = np.array([])
            self.face_areas = np.array([])
            return

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
                    if length > 1e-9:
                        # Outward normal in-plane (z-component is zero)
                        normal = np.array([edge[1], -edge[0], 0.0]) / length
                        self.face_normals[ci, fi] = normal
                elif self.dimension == 3:
                    # Compute polygon normal by triangulating from the face centroid
                    centroid = np.mean(nodes, axis=0)
                    area_vec = np.zeros(3)
                    for k in range(len(nodes)):
                        p1 = nodes[k]
                        p2 = nodes[(k + 1) % len(nodes)]
                        area_vec += np.cross(p1 - centroid, p2 - centroid)
                    area_vec *= 0.5
                    area = np.linalg.norm(area_vec)
                    self.face_areas[ci, fi] = area
                    if area > 1e-9:
                        self.face_normals[ci, fi] = area_vec / area

        # Orient normals to point outwards from the cell centroid
        for ci in range(num_cells):
            for fi in range(len(self.cell_faces[ci])):
                vec_to_face = self.face_midpoints[ci, fi] - self.cell_centroids[ci]
                if np.dot(self.face_normals[ci, fi], vec_to_face) < 0:
                    self.face_normals[ci, fi] *= -1

    def _compute_cell_volumes(self) -> None:
        """
        Compute the volume of each cell.

        For 2D cells, this computes the area using the shoelace formula.
        For 3D cells, it uses the divergence theorem on face contributions.
        """
        self.cell_volumes = np.zeros(self.num_cells)
        if self.dimension == 2:
            # Shoelace formula for polygon area
            for i, conn in enumerate(self.cell_connectivity):
                nodes = self.node_coords[conn][:, :2]
                x, y = nodes[:, 0], nodes[:, 1]
                self.cell_volumes[i] = 0.5 * abs(
                    np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))
                )
        elif self.dimension == 3:
            # Divergence theorem: V = (1/3) * sum(face_mid . face_normal * face_area)
            contrib = (self.face_midpoints * self.face_normals).sum(
                axis=2
            ) * self.face_areas
            self.cell_volumes = np.sum(contrib, axis=1) / 3.0
        else:
            # For other dimensions, volume is not computed
            self.cell_volumes = np.zeros(self.num_cells)

    def print_summary(self) -> None:
        """Prints a nicely formatted summary of the mesh and its quality metrics."""
        if not self._is_analyzed:
            print("Mesh has not been analyzed. Please run analyze_mesh() first.")
            return

        print("\n" + "=" * 80)
        print(f"{ 'Mesh Analysis Report':^80}")
        print("=" * 80)

        # --- General Information ---
        print(f"\n{ '--- General Information ---':^80}\n")
        print(f"  {'Dimension:' :<25} {self.dimension}D")
        print(f"  {'Number of Nodes:' :<25} {self.num_nodes}")
        print(f"  {'Number of Cells:' :<25} {self.num_cells}")

        # --- Geometric Properties ---
        if self.num_nodes > 0:
            x_min, y_min, z_min = np.min(self.node_coords, axis=0)
            x_max, y_max, z_max = np.max(self.node_coords, axis=0)
            print(f"\n{'--- Geometric Properties ---':^80}\n")
            print(f"  {'X Range:':<25} {x_min:.4f} to {x_max:.4f}")
            print(f"  {'Y Range:':<25} {y_min:.4f} to {y_max:.4f}")
            if self.dimension == 3:
                print(f"  {'Z Range:':<25} {z_min:.4f} to {z_max:.4f}")

        # --- Cell Geometry ---
        if self.cell_volumes.size > 0:
            print(f"\n{'--- Cell Geometry ---':^80}\n")

            # Cell Type Information
            cell_type_counts = {}
            for conn in self.cell_connectivity:
                n_sides = len(conn)
                label = f"{n_sides}-gon"
                if n_sides == 3:
                    label = "Triangle"
                elif n_sides == 4:
                    label = "Quadrilateral"
                cell_type_counts[label] = cell_type_counts.get(label, 0) + 1

            print("  Cell Type Distribution:")
            for label, count in cell_type_counts.items():
                print(f"    - {label+':':<20} {count}")

            print(f"\n  {'Metric':<20} {'Min':>15} {'Max':>15} {'Average':>15}")
            print(f"  {'-'*19} {'-'*15} {'-'*15} {'-'*15}")
            vol_min = np.min(self.cell_volumes)
            vol_max = np.max(self.cell_volumes)
            vol_avg = np.mean(self.cell_volumes)
            print(
                f"  {'Cell Volume':<20} {vol_min:>15.4e} {vol_max:>15.4e} {vol_avg:>15.4e}"
            )

        self._compute_quality()
        self.quality.print_summary()

        print("\n" + "=" * 80)

    def _compute_quality(self) -> None:
        """Computes all available mesh quality metrics."""
        self.quality.compute(self)
