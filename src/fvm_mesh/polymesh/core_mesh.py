# -*- coding: utf-8 -*-
"""
Core data structures for unstructured meshes.

This module defines the `CoreMesh` class, which is a fundamental data structure
for representing unstructured meshes. It stores the essential components of a mesh,
including node coordinates, cell connectivity, and boundary information.

Key Features:
- Reading mesh data from Gmsh .msh files.
- Storing node coordinates, cell connectivity, and cell types.
- Handling boundary information from physical groups.
- Computing derived data like cell centroids and neighbors.

The `CoreMesh` class is designed to be a base class for more specialized mesh
representations, providing a solid foundation for mesh analysis and manipulation.
"""

from typing import Dict, List, Tuple, Any
from collections import defaultdict

import gmsh
import numpy as np


class CoreMesh:
    """
    Represents the core mesh information for unstructured meshes.

    This class stores the fundamental components of a mesh, including node
    coordinates, cell connectivity, and boundary information. It provides
    methods to read mesh data from a Gmsh .msh file and to compute basic
    derived data like cell neighbors and centroids.

    Attributes:
        dimension (int): The dimension of the mesh (e.g., 2 for 2D, 3 for 3D).
        num_nodes (int): The total number of nodes in the mesh.
        num_cells (int): The total number of cells (elements) in the mesh.
        node_coords (np.ndarray): An array of shape (N, 3) storing the x, y, z
            coordinates of each node.
        cell_connectivity (List[List[int]]): A list where each inner list contains
            the node indices forming a cell.
        cell_type_ids (np.ndarray): An array storing an integer ID for the type of
            each cell.
        cell_type_map (Dict[int, Dict[str, Any]]): A dictionary mapping cell type
            IDs to their properties (e.g., name, number of nodes).
        cell_neighbors (np.ndarray): An array storing the indices of neighboring
            cells for each cell.
        cell_centroids (np.ndarray): An array storing the computed centroid of each
            cell.
        boundary_faces_nodes (np.ndarray): An array storing the node connectivity
            of faces on the boundary.
        boundary_faces_tags (np.ndarray): An array storing the physical tags
            associated with each boundary face.
        boundary_tag_map (Dict[str, int]): A dictionary mapping physical group
            names to their integer tags.
    """

    def __init__(self) -> None:
        """Initializes the CoreMesh instance."""
        self.dimension: int = 0
        self.num_nodes: int = 0
        self.num_cells: int = 0
        self._is_analyzed: bool = False

        # Geometry/connectivity
        self.node_coords: np.ndarray = np.array([])  # (N, 3)
        self.cell_connectivity: List[List[int]] = []
        self.cell_type_ids: np.ndarray = np.array([])
        self.cell_type_map: Dict[int, Dict[str, Any]] = {}

        # Derived info for adjacency
        self.cell_faces: List[List[List[int]]] = []
        self.cell_neighbors: np.ndarray = np.array([])
        self.cell_centroids: np.ndarray = np.array([])
        self.cell_face_tags: np.ndarray = np.array([])

        # Boundary info
        self.boundary_faces_nodes: np.ndarray = np.array([])  # (M, n_face_nodes)
        self.boundary_faces_tags: np.ndarray = np.array([])
        self.boundary_tag_map: Dict[str, int] = {}

        # Internal Gmsh node tag to index mapping
        self._tag_to_index: Dict[int, int] = {}

    def read_gmsh(self, msh_file: str, gmsh_verbose: int = 0) -> None:
        """
        Reads mesh data from a Gmsh .msh file.

        This method populates the CoreMesh instance with data from the specified
        .msh file, including nodes, cells, and boundary information.

        Args:
            msh_file (str): The path to the .msh file.
            gmsh_verbose (int): The verbosity level for the Gmsh API.
        """
        if gmsh is None:
            raise RuntimeError("Gmsh Python API is not available.")

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
        """Reads node coordinates and creates a tag-to-index mapping."""
        raw_tags, raw_coords, _ = gmsh.model.mesh.getNodes()
        self.node_coords = np.array(raw_coords).reshape(-1, 3)
        self.num_nodes = self.node_coords.shape[0]
        self._tag_to_index = {int(t): i for i, t in enumerate(raw_tags)}

    def _read_elements(self) -> None:
        """Reads cell connectivity and type information from the mesh."""
        elem_types, _, connectivity_list = gmsh.model.mesh.getElements()
        self.dimension = self._get_mesh_dimension(np.array(elem_types))

        all_cell_conn, all_type_ids = [], []
        type_counter = 0
        for i, et in enumerate(elem_types):
            props = gmsh.model.mesh.getElementProperties(et)
            if int(props[1]) != self.dimension:
                continue

            self.cell_type_map[type_counter] = {
                "name": props[0],
                "num_nodes": int(props[3]),
            }
            raw_conn = np.array(connectivity_list[i]).reshape(-1, int(props[3]))
            mapped_conn = np.vectorize(self._tag_to_index.get)(raw_conn).tolist()
            all_cell_conn.extend(mapped_conn)
            all_type_ids.extend([type_counter] * len(mapped_conn))
            type_counter += 1

        if all_cell_conn:
            self.cell_connectivity = all_cell_conn
            self.cell_type_ids = np.array(all_type_ids, dtype=int)
            self.num_cells = len(all_cell_conn)

    def _get_mesh_dimension(self, elem_types: np.ndarray) -> int:
        """Determines the dimension of the mesh from element types."""
        dim = 0
        for et in elem_types:
            props = gmsh.model.mesh.getElementProperties(int(et))
            dim = max(dim, int(props[1]))
        return dim

    def _read_physical_groups(self) -> None:
        """
        Reads physical groups of dimension (dim-1) to set up boundary faces.
        This implementation enforces that only one physical group can be assigned to each face.
        """
        bdim = self.dimension - 1
        if bdim < 0:
            return

        # A dictionary to map unique faces to their node connectivity and associated physical tag.
        # The key is a frozenset of node indices, ensuring uniqueness regardless of node ordering.
        faces_map: Dict[frozenset, Dict[str, Any]] = {}

        physical_groups = gmsh.model.getPhysicalGroups(bdim)
        if not physical_groups:
            return

        for dim, tag in physical_groups:
            name = gmsh.model.getPhysicalName(dim, tag) or str(tag)
            self.boundary_tag_map[name] = tag
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
                        try:
                            face_nodes = [self._tag_to_index[t] for t in face_tags]
                        except KeyError as e:
                            raise RuntimeError(
                                f"Mesh data inconsistency: Node tag {e} from a boundary face "
                                "was not found in the global node list. This may indicate a corrupt mesh file."
                            ) from e

                        face_key = frozenset(face_nodes)

                        if face_key in faces_map:
                            existing_tag = faces_map[face_key]["tag"]
                            if existing_tag != tag:
                                existing_name = gmsh.model.getPhysicalName(
                                    dim, existing_tag
                                ) or str(existing_tag)
                                raise ValueError(
                                    f"Boundary face with nodes {face_nodes} is assigned to multiple physical groups: "
                                    f"'{existing_name}' (tag {existing_tag}) and '{name}' (tag {tag}). "
                                    "Each boundary face can only be assigned to one physical group."
                                )
                        else:
                            faces_map[face_key] = {"nodes": face_nodes, "tag": tag}

        if faces_map:
            self.boundary_faces_nodes = np.array(
                [data["nodes"] for data in faces_map.values()], dtype=int
            )
            self.boundary_faces_tags = np.array(
                [data["tag"] for data in faces_map.values()], dtype=int
            )

    def analyze_mesh(self) -> None:
        """
        Computes derived mesh properties like centroids and neighbors.

        This method sets the `_is_analyzed` flag to True after completion.
        """
        if self.num_cells == 0:
            raise RuntimeError("No cells found. Read a mesh first.")
        if self._is_analyzed:
            return
        self._compute_centroids()
        self._extract_cell_faces()
        self._extract_cell_neighbors()
        self._compute_cell_face_tags()
        self._is_analyzed = True

    def _extract_cell_faces(self) -> None:
        """
        Extracts the faces for each cell in the mesh.
        """
        face_templates = {
            4: [[0, 1, 2], [0, 3, 1], [1, 3, 2], [2, 0, 3]],  # Tet
            8: [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [0, 1, 5, 4],
                [1, 2, 6, 5],
                [2, 3, 7, 6],
                [3, 0, 4, 7],
            ],  # Hex
            6: [
                [0, 1, 2],
                [3, 4, 5],
                [0, 1, 4, 3],
                [1, 2, 5, 4],
                [2, 0, 3, 5],
            ],  # Wedge
        }
        self.cell_faces = [
            self._get_faces_for_cell(conn, face_templates)
            for conn in self.cell_connectivity
        ]

    def _get_faces_for_cell(self, conn: List[int], templates: Dict) -> List[List[int]]:
        """Returns the faces for a single cell connection list."""
        n = len(conn)
        if self.dimension == 2:
            return [[conn[i], conn[(i + 1) % n]] for i in range(n)]
        if self.dimension == 3 and n in templates:
            return [[conn[idx] for idx in face] for face in templates[n]]
        return []

    def _extract_cell_neighbors(self) -> None:
        """
        Extracts cell-to-cell neighbor connectivity.
        """
        if not self.cell_faces:
            self.cell_neighbors = np.array([])
            return

        max_faces = max(len(f) for f in self.cell_faces)
        neighbors = -np.ones((self.num_cells, max_faces), dtype=int)
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

        self.cell_neighbors = neighbors

    def _compute_cell_face_tags(self) -> None:
        """
        Computes tags for each face of each cell.
        Boundary faces get a tag from physical groups.
        Interior faces get a tag of 0.
        """
        if not self.cell_faces:
            return

        # Build a mapping from boundary face nodes to their tags.
        # Using frozenset for order-independent node matching.
        boundary_face_map = {
            frozenset(nodes): tag
            for nodes, tag in zip(self.boundary_faces_nodes, self.boundary_faces_tags)
        }

        max_faces = self.cell_neighbors.shape[1]
        self.cell_face_tags = np.zeros((self.num_cells, max_faces), dtype=np.int32)

        for i in range(self.num_cells):
            for j, neighbor_idx in enumerate(self.cell_neighbors[i]):
                if neighbor_idx == -1:  # This is a boundary face
                    face_nodes = frozenset(self.cell_faces[i][j])
                    if face_nodes in boundary_face_map:
                        self.cell_face_tags[i, j] = boundary_face_map[face_nodes]

    def _compute_centroids(self) -> None:
        """Computes the centroid of each cell."""
        self.cell_centroids = np.array(
            [np.mean(self.node_coords[c], axis=0) for c in self.cell_connectivity]
        )

    def plot(
        self,
        filepath: str = "mesh_plot.png",
        parts: np.ndarray | None = None,
        show_cells: bool = True,
        show_nodes: bool = True,
    ) -> None:
        """
        Plots the mesh and saves it to a file.

        Args:
            filepath (str): The path to save the plot image.
            parts (np.ndarray, optional): An array mapping each cell to a
                partition ID for coloring. Defaults to None.
            show_cells (bool): Whether to display cell outlines.
            show_nodes (bool): Whether to display mesh nodes.
        """
        if self.dimension != 2:
            print("Plotting is currently supported only for 2D meshes.")
            return

        import matplotlib.pyplot as plt
        from ..common.utility import plot_mesh

        fig, ax = plt.subplots(figsize=(10, 8))
        plot_mesh(
            ax,
            self.node_coords[:, :2],
            self.cell_connectivity,
            show_nodes=show_nodes,
            show_cells=show_cells,
            parts=parts,
            title="Mesh Plot",
        )
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Mesh plot saved to: {filepath}")
