import numpy as np
from typing import List

from .core_mesh import CoreMesh


class PolyMesh(CoreMesh):
    """Represents a polygonal mesh with derived geometric information."""

    def __init__(self):
        super().__init__()

        # Analysis flag
        self._is_analyzed: bool = False

        # computed fields
        self.cell_centroids: np.ndarray = np.array([])
        self.cell_volumes: np.ndarray = np.array([])
        self.cell_faces: List[List[List[int]]] = []
        self.face_midpoints: np.ndarray = np.array([])
        self.face_normals: np.ndarray = np.array([])
        self.face_areas: np.ndarray = np.array([])

        # Quality metrics
        self.min_max_volume_ratio: float = 0.0
        self.cell_skewness_values: np.ndarray = np.array([])
        self.cell_non_orthogonality_values: np.ndarray = np.array([])
        self.cell_aspect_ratio_values: np.ndarray = np.array([])
        self.connectivity_issues: List[str] = []

    @classmethod
    def from_gmsh(cls, msh_file: str, gmsh_verbose: int = 0) -> "PolyMesh":
        """Create a PolyMesh instance by reading a Gmsh .msh file."""
        mesh = cls()
        if msh_file:
            mesh.read_gmsh(msh_file, gmsh_verbose)
        return mesh

    def analyze_mesh(self) -> None:
        """Compute centroids, faces, neighbors, face areas/normals, and cell volumes."""
        if self.num_cells == 0:
            raise RuntimeError(
                "No cells available. Call read_gmsh or populate cells first"
            )
        self.compute_centroids()
        self.extract_neighbors()

        self._extract_cell_faces()
        self._compute_face_midpoints_areas_normals()
        self._compute_cell_volumes()
        self._is_analyzed = True  # Set flag after successful analysis

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

        print("\n" + "=" * 80)
