# -*- coding: utf-8 -*-
from typing import Dict, List, Tuple, Any

import gmsh
import numpy as np


class CoreMesh:
    """
    Represents the core mesh information needed for partitioning and reordering.

    This class stores the fundamental components of a mesh, including node
    coordinates, cell connectivity, and boundary information. It provides
    methods to read mesh data from a Gmsh .msh file and to compute basic
    derived data like cell neighbors and centroids.

    This class is not intended to be used directly for simulations but serves
    as a base class for more specialized mesh representations.

    Attributes:
        dimension (int): The dimension of the mesh (e.g., 2 for 2D, 3 for 3D).
        num_nodes (int): The total number of nodes in the mesh.
        num_cells (int): The total number of cells (elements) in the mesh.
        node_coords (np.ndarray): A numpy array of shape (N, 3) storing the
            x, y, z coordinates of each node.
        cell_connectivity (List[List[int]]): A list of lists, where each inner
            list contains the node indices forming a cell.
        cell_type_ids (np.ndarray): A numpy array storing an integer ID for the
            type of each cell.
        cell_type_map (Dict[int, Dict[str, Any]]): A dictionary mapping cell
            type IDs to their properties (e.g., name, number of nodes).
        cell_neighbors (np.ndarray): A numpy array storing the indices of
            neighboring cells for each cell.
        cell_centroids (np.ndarray): A numpy array storing the computed
            centroid of each cell.
        boundary_faces_nodes (np.ndarray): A numpy array storing the node
            connectivity of faces on the boundary.
        boundary_faces_tags (np.ndarray): A numpy array storing the physical
            tags associated with each boundary face.
        boundary_tag_map (Dict[str, int]): A dictionary mapping physical group
            names to their integer tags.
    """

    def __init__(self) -> None:
        # Core mesh attributes
        self.dimension: int = 0
        self.num_nodes: int = 0
        self.num_cells: int = 0

        # Analysis flag
        self._is_analyzed: bool = False

        # Geometry/connectivity
        self.node_coords: np.ndarray = np.array([])  # (N, 3)
        self.cell_connectivity: List[List[int]] = []
        self.cell_type_ids: np.ndarray = np.array([])
        self.cell_type_map: Dict[int, Dict[str, Any]] = {}

        # Derived info for adjacency
        self.cell_neighbors: np.ndarray = np.array([])
        self.cell_centroids: np.ndarray = np.array([])

        # Boundary info
        self.boundary_faces_nodes: np.ndarray = np.array([])  # (M, n_face_nodes)
        self.boundary_faces_tags: np.ndarray = np.array([])
        self.boundary_tag_map: Dict[str, int] = {}

        # Internal Gmsh node tag to index mapping
        self._tag_to_index: Dict[int, int] = {}

    # ---------- I/O ----------

    def read_gmsh(self, msh_file: str, gmsh_verbose: int = 0) -> None:
        """
        Read mesh nodes and elements from a Gmsh .msh file.

        This method populates the CoreMesh instance with data from the specified
        .msh file. It reads node coordinates, cell connectivity, and boundary
        information.

        Args:
            msh_file (str): The path to the .msh file.
            gmsh_verbose (int): The verbosity level for the Gmsh API.
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
            self._tag_to_index = {int(t): i for i, t in enumerate(raw_tags)}

            elem_types, elem_tags_list, connectivity_list = (
                gmsh.model.mesh.getElements()
            )

            # Determine mesh dimension
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
                # Store type info (name, number of nodes)
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

            self._read_gmsh_boundary_groups()

        finally:
            gmsh.finalize()

    def _read_gmsh_boundary_groups(self) -> None:
        """
        Read physical groups at dimension (self.dimension - 1) as boundary faces.

        This method queries Gmsh for physical groups defined on entities of
        dimension one less than the mesh dimension and interprets them as
        boundary faces.
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
            # Ensure all faces have the same node count before vstacking
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

    def analyze_mesh(self) -> None:
        """
        Compute centroids and neighbors.

        This method computes derived mesh properties like cell centroids and
        cell-to-cell neighbor connectivity. It sets the `_is_analyzed` flag
        to True after successful completion.
        """
        if self.num_cells == 0:
            raise RuntimeError(
                "No cells available. Call read_gmsh or populate cells first"
            )
        self._compute_centroids()
        self._extract_neighbors()

        self._is_analyzed = True  # Set flag after successful analysis

    def _extract_neighbors(self) -> None:
        """
        Extract cell-to-cell neighbor connectivity.

        This method determines which cells are adjacent to each other by
        identifying shared faces. The results are stored in the
        `cell_neighbors` attribute.
        """
        # Face templates for common element node counts (3D); 2D handled generically
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
        cell_faces = []
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
            cell_faces.append(faces)

        if not cell_faces:
            self.cell_neighbors = np.array([])
            return

        max_faces = max(len(f) for f in cell_faces)
        num = self.num_cells
        neighbors = -np.ones((num, max_faces), dtype=int)
        face_map: Dict[Tuple[int, ...], List[int]] = {}
        for ci, faces in enumerate(cell_faces):
            for face in faces:
                key = tuple(sorted(face))
                face_map.setdefault(key, []).append(ci)
        for ci, faces in enumerate(cell_faces):
            for fi, face in enumerate(faces):
                key = tuple(sorted(face))
                elems = face_map.get(key, [])
                if len(elems) == 2:
                    neighbors[ci, fi] = elems[0] if elems[1] == ci else elems[1]
                else:
                    neighbors[ci, fi] = -1

        self.cell_neighbors = neighbors

    def _compute_centroids(self) -> None:
        """Compute the centroid of each cell."""
        self.cell_centroids = np.array(
            [np.mean(self.node_coords[c], axis=0) for c in self.cell_connectivity]
        )

    def plot(self, filepath: str = "mesh_plot.png", parts: np.ndarray | None = None):
        """
        Plot the mesh and save it to a file.

        This method generates a 2D plot of the mesh, showing nodes and cells.
        If partition information is provided, cells are colored accordingly.

        Args:
            filepath (str): The path to save the plot image file.
            parts (np.ndarray, optional): An array mapping each cell to a
                partition ID for coloring. Defaults to None.
        """
        if self.dimension != 2:
            print("Plotting is currently supported only for 2D meshes.")
            return

        import matplotlib.pyplot as plt
        from common.utility import plot_mesh

        fig, ax = plt.subplots(figsize=(10, 8))

        plot_mesh(
            ax,
            self.node_coords[:, :2],
            self.cell_connectivity,
            show_nodes=True,
            show_cells=True,
            parts=parts,
            title="Mesh Plot",
        )

        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Mesh plot saved to: {filepath}")
