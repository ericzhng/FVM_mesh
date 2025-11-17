# -*- coding: utf-8 -*-
"""
This module defines the PolyMesh class, a comprehensive data structure for
representing 2D and 3D unstructured polygonal meshes. It is designed to be a
stand-alone, data-centric container for mesh information required for finite
volume method (FVM) solvers.

The PolyMesh class handles reading mesh data from Gmsh .msh files, computes
all necessary geometric and topological properties, and provides utilities for
analysis and visualization.
"""

from typing import Any, Dict, List, Tuple
from collections import Counter

import numpy as np
import gmsh
import matplotlib.pyplot as plt

from .quality import MeshQuality
from .reporting import format_quality_summary


class PolyMesh:
    """
    A data-centric class for storing and analyzing unstructured polygonal meshes.

    This class holds all the fundamental mesh data, including topology and
    computed geometric properties. It is designed to be initialized from a Gmsh
    .msh file and subsequently analyzed to prepare it for numerical simulations.

    Attributes:
        dimension (int): The spatial dimension of the mesh (2 or 3).
        n_nodes (int): The total number of nodes (vertices) in the mesh.
        n_cells (int): The total number of cells (elements) in the mesh.
        node_coords (np.ndarray): Node coordinates.
            - Shape: `(n_nodes, 3)`
            - `dtype`: `float`
        cell_node_connectivity (List[List[int]]): A jagged list where each inner
            list contains the node indices for a single cell.
            - Shape: `(n_cells, n_nodes_per_cell)`
        cell_element_types (np.ndarray): An integer ID for the Gmsh type of each cell.
            - Shape: `(n_cells,)`
            - `dtype`: `int`
        element_type_properties (Dict[int, Dict[str, Any]]): A map from the integer ID
            in `cell_element_types` to properties of that cell type, such as its
            name and number of nodes.
        cell_face_nodes (List[List[List[int]]]): A jagged list where each inner list
            contains the faces for a single cell. Each face is a list of node
            indices.
            - Shape: `(n_cells, n_faces_per_cell, n_nodes_per_face)`
        cell_neighbors (np.ndarray): Indices of neighboring cells for each face
            of each cell. A value of -1 indicates a boundary face.
            - Shape: `(n_cells, max_faces_per_cell)`
            - `dtype`: `int`
        cell_face_tags (np.ndarray): Physical tag for each face of each cell.
            For boundary faces, this is the tag from the physical group. For
            interior faces, it is 0.
            - Shape: `(n_cells, max_faces_per_cell)`
            - `dtype`: `int`
        cell_centroids (np.ndarray): The geometric center of each cell.
            - Shape: `(n_cells, 3)`
            - `dtype`: `float`
        cell_volumes (np.ndarray): The volume (3D) or area (2D) of each cell.
            - Shape: `(n_cells,)`
            - `dtype`: `float`
        cell_face_midpoints (np.ndarray): The geometric center of each face of each
            cell.
            - Shape: `(n_cells, max_faces_per_cell, 3)`
            - `dtype`: `float`
        cell_face_normals (np.ndarray): The normal vector of each face, oriented to
            point outwards from its parent cell.
            - Shape: `(n_cells, max_faces_per_cell, 3)`
            - `dtype`: `float`
        cell_face_areas (np.ndarray): The area (3D) or length (2D) of each face.
            - Shape: `(n_cells, max_faces_per_cell)`
            - `dtype`: `float`
        face_to_centroid_distances (np.ndarray): The distance from each face's midpoint
            to its parent cell's centroid.
            - Shape: `(n_cells, max_faces_per_cell)`
            - `dtype`: `float`
        boundary_face_nodes (np.ndarray): Node indices for each unique boundary face.
            - Shape: `(n_boundary_faces, n_nodes_per_face)`
            - `dtype`: `int`
        boundary_face_tags (np.ndarray): The physical tag for each unique boundary face.
            - Shape: `(n_boundary_faces,)`
            - `dtype`: `int`
        boundary_patch_map (Dict[str, int]): A mapping from physical group names
            (e.g., "inlet", "wall") to their integer tags.
        quality (MeshQuality): An object for computing and storing mesh
            quality metrics.
    """

    def __init__(self) -> None:
        """Initializes the PolyMesh instance with empty attributes."""
        # Core Mesh Properties
        self.dimension: int = 0
        self.n_nodes: int = 0
        self.n_cells: int = 0
        self._is_analyzed: bool = False

        # Topology Data (read or defined)
        self.node_coords: np.ndarray = np.array([])
        self.cell_node_connectivity: List[List[int]] = []
        self.cell_element_types: np.ndarray = np.array([])
        self.element_type_properties: Dict[int, Dict[str, Any]] = {}

        # Topology Data (derived)
        self.cell_face_nodes: List[List[List[int]]] = []
        self.cell_neighbors: np.ndarray = np.array([])
        self.cell_face_tags: np.ndarray = np.array([])

        # Boundary Data
        self.boundary_face_nodes: np.ndarray = np.array([])
        self.boundary_face_tags: np.ndarray = np.array([])
        self.boundary_patch_map: Dict[str, int] = {}

        # Computed Geometric Properties
        self.cell_centroids: np.ndarray = np.array([])
        self.cell_volumes: np.ndarray = np.array([])
        self.cell_face_midpoints: np.ndarray = np.array([])
        self.cell_face_normals: np.ndarray = np.array([])
        self.cell_face_areas: np.ndarray = np.array([])
        self.face_to_centroid_distances: np.ndarray = np.array([])

        # Internal Data
        self._tag_to_index: Dict[int, int] = {}

        # Quality Metrics
        self.quality: MeshQuality | None = None

    # =========================================================================
    # Public API
    # =========================================================================

    @classmethod
    def from_gmsh(cls, msh_file: str, gmsh_verbose: int = 0) -> "PolyMesh":
        """
        Creates and returns a PolyMesh instance from a Gmsh .msh file.

        This is the recommended factory method for creating a mesh.

        Args:
            msh_file (str): The path to the .msh file.
            gmsh_verbose (int): The verbosity level for the Gmsh API (0-10).

        Returns:
            A new PolyMesh instance populated with data from the file.
        """
        mesh = cls()
        mesh.read_gmsh(msh_file, gmsh_verbose)
        return mesh

    @classmethod
    def create_structured_quad_mesh(cls, nx: int, ny: int) -> "PolyMesh":
        """
        Creates a structured quadrilateral mesh of size nx x ny with tagged boundaries.

        This factory method is useful for creating simple meshes for testing
        without needing a mesh file. The boundaries are tagged as:
        - 1: bottom
        - 2: right
        - 3: top
        - 4: left

        Args:
            nx (int): Number of cells in the x-direction.
            ny (int): Number of cells in the y-direction.

        Returns:
            A new, analyzed PolyMesh instance.
        """
        mesh = cls()
        mesh.dimension = 2

        num_nodes_x = nx + 1
        num_nodes_y = ny + 1
        mesh.n_nodes = num_nodes_x * num_nodes_y
        mesh.n_cells = nx * ny

        # Generate node coordinates
        node_coords = []
        for j in range(num_nodes_y):
            for i in range(num_nodes_x):
                node_coords.append([float(i), float(j), 0.0])
        mesh.node_coords = np.array(node_coords)

        # Generate cell connectivity
        cell_connectivity = []
        for j in range(ny):
            for i in range(nx):
                n0 = j * num_nodes_x + i
                n1 = j * num_nodes_x + (i + 1)
                n2 = (j + 1) * num_nodes_x + (i + 1)
                n3 = (j + 1) * num_nodes_x + i
                cell_connectivity.append([n0, n1, n2, n3])
        mesh.cell_node_connectivity = cell_connectivity

        # Add element type information for quads (Gmsh type 3)
        mesh.cell_element_types = np.full(mesh.n_cells, 0, dtype=int)
        mesh.element_type_properties[0] = {"name": "Quad 4", "num_nodes": 4}

        # --- Find and tag boundary faces ---
        face_to_cells: Dict[Tuple[int, ...], List[int]] = {}
        temp_cell_faces = [
            [[conn[i], conn[(i + 1) % 4]] for i in range(4)]
            for conn in mesh.cell_node_connectivity
        ]

        for cell_idx, faces in enumerate(temp_cell_faces):
            for face in faces:
                key = tuple(sorted(face))
                face_to_cells.setdefault(key, []).append(cell_idx)

        boundary_faces_nodes = []
        boundary_faces_tags = []

        bottom_tag, right_tag, top_tag, left_tag = 1, 2, 3, 4

        for face_nodes_tuple, cells in face_to_cells.items():
            if len(cells) == 1:
                face_nodes = list(face_nodes_tuple)
                node1_coords = mesh.node_coords[face_nodes[0]]
                node2_coords = mesh.node_coords[face_nodes[1]]

                # Check for boundary
                if np.allclose(node1_coords[1], 0) and np.allclose(node2_coords[1], 0):
                    tag = bottom_tag
                elif np.allclose(node1_coords[0], nx) and np.allclose(
                    node2_coords[0], nx
                ):
                    tag = right_tag
                elif np.allclose(node1_coords[1], ny) and np.allclose(
                    node2_coords[1], ny
                ):
                    tag = top_tag
                elif np.allclose(node1_coords[0], 0) and np.allclose(
                    node2_coords[0], 0
                ):
                    tag = left_tag
                else:
                    continue

                boundary_faces_nodes.append(face_nodes)
                boundary_faces_tags.append(tag)

        mesh.boundary_face_nodes = np.array(boundary_faces_nodes, dtype=int)
        mesh.boundary_face_tags = np.array(boundary_faces_tags, dtype=int)
        mesh.boundary_patch_map = {
            "bottom": bottom_tag,
            "right": right_tag,
            "top": top_tag,
            "left": left_tag,
        }

        # Analyze the mesh to compute all derived properties
        mesh.analyze_mesh()

        return mesh

    def analyze_mesh(self) -> None:
        """
        Computes all derived topological and geometric properties for the mesh.

        This method orchestrates a series of computations to build the full
        mesh data structure. It should be called after reading the mesh file.
        """
        if self.n_cells == 0 or self.n_nodes == 0:
            raise RuntimeError(
                "Mesh has no cells or nodes. Load a mesh with read_gmsh() before analyzing."
            )
        if self._is_analyzed:
            return

        # 1. Compute basic topology and connectivity
        self._extract_cell_faces()
        self._compute_face_topology()

        # 2. Compute geometric properties
        self._compute_cell_centroids()
        self._compute_face_properties()  # Includes areas, normals, midpoints
        self._compute_face_to_centroid_dist()
        self._compute_cell_volumes()

        self._is_analyzed = True

    def print_summary(self) -> None:
        """Prints a formatted summary report of the mesh analysis."""
        if not self._is_analyzed:
            print("Mesh not analyzed. Run analyze_mesh() first.")
            return

        print("\n" + "=" * 80)
        print(f"{"Mesh Analysis Report":^80}")
        print("=" * 80)
        self._print_general_info()
        self._print_geometric_properties()
        self._print_cell_geometry()

        # Compute quality metrics on demand if not already done
        if self.quality is None:
            self.quality = MeshQuality.from_mesh(self)

        print(format_quality_summary(self.quality))
        print("\n" + "=" * 80)

    def plot(
        self,
        filepath: str = "mesh_plot.png",
        parts: np.ndarray | None = None,
        show_cells: bool = True,
        show_nodes: bool = True,
    ) -> None:
        """
        Generates a 2D plot of the mesh and saves it to a file.

        Note: Plotting is currently only supported for 2D meshes.

        Args:
            filepath (str): The path to save the plot image.
            parts (np.ndarray, optional): An array mapping each cell to a
                partition ID for coloring. Shape: (n_cells,), dtype: int.
            show_cells (bool): Whether to display cell outlines.
            show_nodes (bool): Whether to display mesh nodes.
        """
        if self.dimension != 2:
            print("Warning: Plotting is currently supported only for 2D meshes.")
            return

        from ..common.utility import plot_mesh

        fig, ax = plt.subplots(figsize=(10, 8))
        plot_mesh(
            ax,
            self.node_coords,
            self.cell_node_connectivity,
            show_nodes=show_nodes,
            show_cells=show_cells,
            parts=parts,
            title="Mesh Plot",
        )
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Mesh plot saved to: {filepath}")

    # =========================================================================
    # Mesh I/O Methods
    # =========================================================================

    def read_gmsh(self, msh_file: str, gmsh_verbose: int = 0) -> None:
        """
        Reads mesh data from a Gmsh .msh file using the Gmsh Python API.

        Args:
            msh_file (str): The path to the .msh file.
            gmsh_verbose (int): The verbosity level for the Gmsh API.
        """
        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", gmsh_verbose)
        try:
            gmsh.open(msh_file)
            self._read_nodes()
            self._read_elements()
            self._read_physical_groups()
        finally:
            gmsh.finalize()

    def _read_nodes(self) -> None:
        """Reads node coordinates and creates the node tag-to-index map."""
        raw_tags, raw_coords, _ = gmsh.model.mesh.getNodes()
        self.node_coords = np.array(raw_coords).reshape(-1, 3)
        self.n_nodes = self.node_coords.shape[0]
        self._tag_to_index = {int(t): i for i, t in enumerate(raw_tags)}

    def _read_elements(self) -> None:
        """Reads element (cell) connectivity and type information."""
        elem_types, _, connectivity_list = gmsh.model.mesh.getElements()
        self.dimension = self._get_mesh_dimension(np.array(elem_types))

        all_cell_conn, all_type_ids = [], []
        for i, et in enumerate(elem_types):
            props = gmsh.model.mesh.getElementProperties(et)
            if int(props[1]) != self.dimension:
                continue  # Skip elements not of the mesh's primary dimension

            self.element_type_properties[et] = {
                "name": props[0],
                "num_nodes": int(props[3]),
            }
            raw_conn = np.array(connectivity_list[i]).reshape(-1, int(props[3]))
            # Vectorized mapping from Gmsh tags to our zero-based indices
            mapped_conn = np.vectorize(self._tag_to_index.get)(raw_conn).tolist()
            all_cell_conn.extend(mapped_conn)
            all_type_ids.extend([et] * len(mapped_conn))

        if all_cell_conn:
            self.cell_node_connectivity = all_cell_conn
            self.cell_element_types = np.array(all_type_ids, dtype=int)
            self.n_cells = len(all_cell_conn)

    def _read_physical_groups(self) -> None:
        """
        Reads physical groups of dimension (dim-1) to define boundary faces.

        This method enforces that each boundary face can only belong to one
        physical group to prevent ambiguity.
        """
        bdim = self.dimension - 1
        if bdim < 0:
            return

        # Use a map to ensure each unique face is processed only once.
        # The key is a frozenset of node indices for order-invariance.
        faces_map: Dict[frozenset, Dict[str, Any]] = {}
        physical_groups = gmsh.model.getPhysicalGroups(bdim)

        for dim, tag in physical_groups:
            name = gmsh.model.getPhysicalName(dim, tag) or str(tag)
            self.boundary_patch_map[name] = tag
            entities = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)

            for ent in entities:
                try:
                    elem_types, _, node_tags_per_type = gmsh.model.mesh.getElements(
                        dim, ent
                    )
                except ValueError:
                    continue  # No elements in this entity

                for i, etype in enumerate(elem_types):
                    props = gmsh.model.mesh.getElementProperties(etype)
                    n_nodes = props[3]
                    raw_conn = node_tags_per_type[i]
                    if len(raw_conn) == 0:
                        continue

                    faces = np.array(raw_conn).reshape(-1, n_nodes)

                    for face_tags in faces:
                        face_nodes = [self._tag_to_index[t] for t in face_tags]
                        face_key = frozenset(face_nodes)
                        if face_key in faces_map and faces_map[face_key]["tag"] != tag:
                            raise ValueError(
                                f"Boundary face with nodes {face_nodes} is assigned to multiple physical groups."
                            )
                        faces_map[face_key] = {"nodes": face_nodes, "tag": tag}

        if faces_map:
            self.boundary_face_nodes = np.array(
                [data["nodes"] for data in faces_map.values()], dtype=int
            )
            self.boundary_face_tags = np.array(
                [data["tag"] for data in faces_map.values()], dtype=int
            )

    # =========================================================================
    # Topology and Connectivity Computations
    # =========================================================================

    def _extract_cell_faces(self) -> None:
        """
        Extracts the faces for each cell based on its connectivity and dimension.
        """
        # Templates for face node ordering in standard 3D cell types.
        face_templates = {
            4: [[0, 1, 2], [0, 3, 1], [1, 3, 2], [2, 0, 3]],  # Tetrahedron
            8: [  # Hexahedron
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [0, 1, 5, 4],
                [1, 2, 6, 5],
                [2, 3, 7, 6],
                [3, 0, 4, 7],
            ],
            6: [  # Wedge
                [0, 1, 2],
                [3, 4, 5],
                [0, 1, 4, 3],
                [1, 2, 5, 4],
                [2, 0, 3, 5],
            ],
        }
        self.cell_face_nodes = [
            self._get_faces_for_cell(conn, face_templates)
            for conn in self.cell_node_connectivity
        ]

    def _get_faces_for_cell(self, conn: List[int], templates: Dict) -> List[List[int]]:
        """Helper to get faces for a single cell."""
        num_nodes = len(conn)
        if self.dimension == 2:
            # For 2D cells, faces are edges.
            return [[conn[i], conn[(i + 1) % num_nodes]] for i in range(num_nodes)]
        if self.dimension == 3 and num_nodes in templates:
            # For 3D cells, use predefined templates.
            return [[conn[idx] for idx in face] for face in templates[num_nodes]]
        return []

    def _compute_face_topology(self) -> None:
        """
        Computes cell-to-cell neighbors and tags for all cell faces.

        This optimized method iterates through faces once to build a lookup map,
        then iterates a second time to populate both the neighbor and tag arrays,
        reducing total loops over the mesh data.
        """
        if not self.cell_face_nodes:
            return

        # --- 1. Build lookup maps ---
        # Map unique faces (as sorted node tuples) to the cells they belong to.
        face_to_cell_map: Dict[Tuple[int, ...], List[int]] = {}
        for cell_idx, faces in enumerate(self.cell_face_nodes):
            for face in faces:
                key = tuple(sorted(face))
                face_to_cell_map.setdefault(key, []).append(cell_idx)

        # Map boundary faces (as frozensets) to their physical tags.
        boundary_face_map = {
            frozenset(nodes): tag
            for nodes, tag in zip(self.boundary_face_nodes, self.boundary_face_tags)
        }

        # --- 2. Initialize arrays and compute topology ---
        max_faces = (
            max(len(f) for f in self.cell_face_nodes) if self.cell_face_nodes else 0
        )
        self.cell_neighbors = -np.ones((self.n_cells, max_faces), dtype=int)
        self.cell_face_tags = np.zeros((self.n_cells, max_faces), dtype=np.int32)

        for cell_idx, faces in enumerate(self.cell_face_nodes):
            for face_idx, face in enumerate(faces):
                key = tuple(sorted(face))
                shared_cells = face_to_cell_map.get(key, [])

                if len(shared_cells) == 2:  # This is an interior face
                    neighbor_idx = (
                        shared_cells[0]
                        if shared_cells[1] == cell_idx
                        else shared_cells[1]
                    )
                    self.cell_neighbors[cell_idx, face_idx] = neighbor_idx

                elif len(shared_cells) == 1:  # This is a boundary face
                    # Neighbor is already -1. Now find the tag.
                    face_key_frozenset = frozenset(face)
                    if face_key_frozenset in boundary_face_map:
                        self.cell_face_tags[cell_idx, face_idx] = boundary_face_map[
                            face_key_frozenset
                        ]

    # =========================================================================
    # Geometric Property Computations
    # =========================================================================

    def _compute_cell_centroids(self) -> None:
        """Computes the geometric centroid of each cell."""
        self.cell_centroids = np.array(
            [
                np.mean(self.node_coords[conn], axis=0)
                for conn in self.cell_node_connectivity
            ]
        )

    def _compute_face_properties(self) -> None:
        """
        Computes geometric properties (midpoint, area, normal) for each face
        of each cell.
        """
        if self.cell_neighbors.size == 0:
            return

        max_faces = self.cell_neighbors.shape[1]
        self.cell_face_midpoints = np.zeros((self.n_cells, max_faces, 3))
        self.cell_face_normals = np.zeros((self.n_cells, max_faces, 3))
        self.cell_face_areas = np.zeros((self.n_cells, max_faces))

        for ci, faces in enumerate(self.cell_face_nodes):
            for fi, face_nodes in enumerate(faces):
                nodes = self.node_coords[face_nodes]
                self.cell_face_midpoints[ci, fi] = np.mean(nodes, axis=0)
                if self.dimension == 2:
                    self._compute_2d_face_metrics(ci, fi, nodes)
                elif self.dimension == 3:
                    self._compute_3d_face_metrics(ci, fi, nodes)

        self._orient_face_normals()

    def _compute_2d_face_metrics(self, ci: int, fi: int, nodes: np.ndarray) -> None:
        """Computes area (length) and normal for a 2D face (edge)."""
        edge_vec = nodes[1] - nodes[0]
        length = np.linalg.norm(edge_vec)
        self.cell_face_areas[ci, fi] = length
        if length > 1e-12:
            # Perpendicular vector in 2D plane
            self.cell_face_normals[ci, fi] = (
                np.array([edge_vec[1], -edge_vec[0], 0.0]) / length
            )

    def _compute_3d_face_metrics(self, ci: int, fi: int, nodes: np.ndarray) -> None:
        """Computes area and normal for a 3D face (polygon)."""
        centroid = np.mean(nodes, axis=0)
        # Use cross product of triangle segments to get area vector
        area_vec = (
            sum(
                np.cross(nodes[k] - centroid, nodes[(k + 1) % len(nodes)] - centroid)
                for k in range(len(nodes))
            )
            / 2.0
        )
        area = np.linalg.norm(area_vec)
        self.cell_face_areas[ci, fi] = area
        if area > 1e-12:
            self.cell_face_normals[ci, fi] = area_vec / area

    def _orient_face_normals(self) -> None:
        """Ensures all face normals point outwards from their cell centroid."""
        for ci in range(self.n_cells):
            for fi in range(len(self.cell_face_nodes[ci])):
                vec_to_face = self.cell_face_midpoints[ci, fi] - self.cell_centroids[ci]
                if np.dot(self.cell_face_normals[ci, fi], vec_to_face) < 0:
                    # Flip the normal if it points inwards
                    self.cell_face_normals[ci, fi] *= -1

    def _compute_face_to_centroid_dist(self) -> None:
        """Computes the distance from each face's midpoint to the cell's centroid."""
        if self.cell_face_midpoints.size == 0:
            return

        # This can be vectorized, but a loop is clearer and sufficient for now.
        self.face_to_centroid_distances = np.zeros_like(self.cell_face_areas)
        for ci in range(self.n_cells):
            for fi in range(len(self.cell_face_nodes[ci])):
                vec = self.cell_face_midpoints[ci, fi] - self.cell_centroids[ci]
                self.face_to_centroid_distances[ci, fi] = np.linalg.norm(vec)

    def _compute_cell_volumes(self) -> None:
        """Computes the volume (3D) or area (2D) of each cell."""
        if self.dimension == 2:
            self._compute_2d_cell_volumes()
        elif self.dimension == 3:
            self._compute_3d_cell_volumes()
        else:
            self.cell_volumes = np.zeros(self.n_cells)

    def _compute_2d_cell_volumes(self) -> None:
        """Computes cell areas for a 2D mesh using the shoelace formula."""
        volumes = np.zeros(self.n_cells)
        for i, conn in enumerate(self.cell_node_connectivity):
            nodes = self.node_coords[conn][:, :2]
            x, y = nodes[:, 0], nodes[:, 1]
            volumes[i] = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        self.cell_volumes = volumes

    def _compute_3d_cell_volumes(self) -> None:
        """
        Computes cell volumes for a 3D mesh using the divergence theorem.
        The volume is calculated as (1/3) * sum(face_midpoint . face_normal * face_area).
        """
        # Dot product of face midpoints and normals, summed over xyz components
        contrib = np.sum(self.cell_face_midpoints * self.cell_face_normals, axis=2)
        # Sum contributions from all faces of each cell
        self.cell_volumes = np.sum(contrib * self.cell_face_areas, axis=1) / 3.0

    # =========================================================================
    # Helper and Utility Methods
    # =========================================================================

    def _get_mesh_dimension(self, elem_types: np.ndarray) -> int:
        """Determines the mesh dimension by finding the max dim of elements."""
        max_dim = 0
        for et in elem_types:
            props = gmsh.model.mesh.getElementProperties(int(et))
            max_dim = max(max_dim, int(props[1]))
        return max_dim

    def _print_general_info(self) -> None:
        """Prints general information about the mesh."""
        print(f"\n{'--- General Information ---':^80}\n")
        print(f"  {'Dimension:':<25} {self.dimension}D")
        print(f"  {'Number of Nodes:':<25} {self.n_nodes}")
        print(f"  {'Number of Cells:':<25} {self.n_cells}")

    def _print_geometric_properties(self) -> None:
        """Prints the geometric bounds of the mesh."""
        if self.n_nodes == 0:
            return
        min_coords = np.min(self.node_coords, axis=0)
        max_coords = np.max(self.node_coords, axis=0)
        print(f"\n{'--- Geometric Bounding Box ---':^80}\n")
        print(f"  {'X Range:':<25} {min_coords[0]:.4f} to {max_coords[0]:.4f}")
        print(f"  {'Y Range:':<25} {min_coords[1]:.4f} to {max_coords[1]:.4f}")
        if self.dimension == 3:
            print(f"  {'Z Range:':<25} {min_coords[2]:.4f} to {max_coords[2]:.4f}")

    def _print_cell_geometry(self) -> None:
        """Prints statistics about cell geometry."""
        if self.cell_volumes.size == 0:
            return
        print(f"\n{'--- Cell Geometry ---':^80}\n")
        self._print_cell_type_distribution()
        print(f"\n  {'Metric':<20} {'Min':>15} {'Max':>15} {'Average':>15}")
        print(f"  {'-'*19} {'-'*15} {'-'*15} {'-'*15}")
        self._print_stat_line("Cell Volume", self.cell_volumes)
        self._print_stat_line("Face-to-Centroid Dist", self.face_to_centroid_distances)

    def _print_cell_type_distribution(self) -> None:
        """Prints the distribution of different cell types using explicit element data."""
        if self.cell_element_types.size == 0:
            return

        type_counts = Counter(self.cell_element_types)

        print("  Cell Type Distribution:")
        for type_id, count in sorted(type_counts.items()):
            type_name = self.element_type_properties.get(type_id, {}).get(
                "name", "Unknown"
            )
            print(f"    - {type_name+':':<20} {count}")

    def _print_stat_line(self, name: str, data: np.ndarray) -> None:
        """Helper to print a formatted statistics line for a given dataset."""
        if data.size == 0:
            return
        # Filter out zeros or invalid values for metrics like distance
        valid_data = data[data > 0]
        if valid_data.size == 0:
            return
        min_val, max_val, avg_val = (
            np.min(valid_data),
            np.max(valid_data),
            np.mean(valid_data),
        )
        print(f"  {name:<25} {min_val:>15.4e} {max_val:>15.4e} {avg_val:>15.4e}")
