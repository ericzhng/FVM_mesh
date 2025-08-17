import numpy as np
from typing import Dict, List, Tuple, Any

import gmsh
import matplotlib.pyplot as plt  # New import for 2D plotting
import pyvista as pv  # New import for 3D plotting


class Mesh:
    """Represents a geometric mesh and computes FVM geometric/connectivity data.

    Responsibilities:
    - read mesh from gmsh (.msh)
    - compute cell centroids, faces, neighbors, face areas/normals, cell volumes
    - provide mesh_data export for solvers

    Note: This class DOES NOT perform partitioning or node renumbering. Those are
    implemented in PartitionManager.
    """

    def __init__(self) -> None:
        # core mesh attributes
        self.dimension: int = 0
        self.num_nodes: int = 0
        self.num_cells: int = 0

        # geometry/connectivity
        self.node_coords: np.ndarray = np.array([])  # (N,3)
        self.cell_connectivity: List[List[int]] = []
        self.cell_type_ids: np.ndarray = np.array([])
        self.cell_type_map: Dict[int, Dict[str, Any]] = {}

        # boundary info
        self.boundary_faces_nodes: np.ndarray = np.array([])  # (M, n_face_nodes)
        self.boundary_faces_tags: np.ndarray = np.array([])
        self.boundary_tag_map: Dict[str, int] = {}

        # computed fields
        self.cell_centroids: np.ndarray = np.array([])
        self.cell_volumes: np.ndarray = np.array([])
        self.cell_faces: List[List[List[int]]] = []
        self.face_midpoints: np.ndarray = np.array([])
        self.face_normals: np.ndarray = np.array([])
        self.face_areas: np.ndarray = np.array([])
        self.cell_neighbors: np.ndarray = np.array([])

        # Analysis flag
        self._is_analyzed: bool = False  # New flag

        # Internal gmsh node tag to index mapping
        self._tag_to_index: Dict[int, int] = {}

    # ---------- I/O ----------
    def read_gmsh(self, msh_file: str, gmsh_verbose: int = 0) -> None:
        """Read mesh nodes and elements from Gmsh .msh file.

        This populates node_coords, cell_connectivity, cell_type_map, and
        boundary face lists (if physical groups exist).
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
            # self._gmsh_node_tags = np.array(raw_tags) # Not strictly needed as an attribute
            self._tag_to_index = {int(t): i for i, t in enumerate(raw_tags)}

            elem_types, elem_tags_list, connectivity_list = (
                gmsh.model.mesh.getElements()
            )

            # determine mesh dimension
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
                # store type info (name, number of nodes)
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

            # read boundary physical groups
            self._read_gmsh_boundary_groups()
        finally:
            gmsh.finalize()

    def _read_gmsh_boundary_groups(self) -> None:
        """Read physical groups at dimension (self.dimension - 1) as boundary faces.

        Notes:
        - We filter physical groups by desired boundary dimension to avoid relying on API
          providing a 'dim' argument.
        - If physical groups contain faces of different element node counts we currently
          store them in separate contiguous blocks (vstack requires consistent number of
          nodes per face type) — mixed face types across groups may not vstack.
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
            # Ensure all faces have same node count before vstacking
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

    # ---------- Analysis ----------
    def analyze_mesh(self) -> None:
        """Compute centroids, faces, neighbors, face areas/normals, and cell volumes.

        This method is independent of any renumbering or partitioning and can be
        called at any time after the mesh is loaded.
        """
        if self.num_cells == 0:
            raise RuntimeError(
                "No cells available. Call read_gmsh or populate cells first"
            )
        self._compute_centroids()
        self._extract_cell_faces()
        self._compute_cell_neighbors()
        self._compute_face_midpoints_areas_normals()
        self._compute_cell_volumes()
        self._is_analyzed = True  # Set flag after successful analysis

    def _compute_centroids(self) -> None:
        self.cell_centroids = np.array(
            [np.mean(self.node_coords[c], axis=0) for c in self.cell_connectivity]
        )

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

    def _compute_cell_neighbors(self) -> None:
        if not self.cell_faces:
            self.cell_neighbors = np.array([])
            return
        max_faces = max(len(f) for f in self.cell_faces)
        num = self.num_cells
        neighbors = -np.ones((num, max_faces), dtype=int)
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
                else:
                    neighbors[ci, fi] = -1
        self.cell_neighbors = neighbors

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

    def get_mesh_data(self) -> Dict[str, Any]:
        """Return structured data useful for FVM solver or partition writer."""
        return {
            "dimension": int(self.dimension),
            "node_coords": self.node_coords,
            "cell_connectivity": self.cell_connectivity,
            "cell_type_ids": self.cell_type_ids,
            "cell_volumes": self.cell_volumes,
            "cell_centroids": self.cell_centroids,
            "face_areas": self.face_areas,
            "face_normals": self.face_normals,
            "cell_neighbors": self.cell_neighbors,
            "boundary_faces_nodes": self.boundary_faces_nodes,
            "boundary_faces_tags": self.boundary_faces_tags,
            "boundary_tag_map": self.boundary_tag_map,
        }

    def plot_mesh(
        self, max_elements_for_labels: int = 50, show_plot: bool = True
    ) -> None:
        """Plots the mesh.

        If analyze_mesh has been run, plots with surface normals, element tags, and areas.
        If the mesh has too many elements (controlled by max_elements_for_labels),
        labels are skipped.

        Args:
            max_elements_for_labels: Maximum number of elements to display labels for.
            show_plot: Whether to display the plot immediately.
        """
        if self.num_cells == 0:
            print("No mesh data to plot. Load a mesh first.")
            return

        plot_labels = self._is_analyzed and self.num_cells <= max_elements_for_labels
        plot_normals = self._is_analyzed
        plot_element_tags = (
            self._is_analyzed and self.num_cells <= max_elements_for_labels
        )
        plot_areas = self._is_analyzed and self.num_cells <= max_elements_for_labels

        if self.dimension == 2:
            self._plot_2d_mesh(
                plot_labels, plot_normals, plot_element_tags, plot_areas, show_plot
            )
        elif self.dimension == 3:
            self._plot_3d_mesh(
                plot_labels, plot_normals, plot_element_tags, plot_areas, show_plot
            )
        else:
            print(f"Plotting not supported for mesh dimension {self.dimension}")

    def _plot_2d_mesh(
        self,
        plot_labels: bool,
        plot_normals: bool,
        plot_element_tags: bool,
        plot_areas: bool,
        show_plot: bool,
    ) -> None:
        """Helper to plot 2D meshes."""
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("2D Mesh Plot")

        # Plot nodes
        ax.plot(
            self.node_coords[:, 0],
            self.node_coords[:, 1],
            "o",
            markersize=2,
            color="blue",
            label="Nodes",
        )

        # Plot cells and their properties
        for i, cell_conn in enumerate(self.cell_connectivity):
            nodes = self.node_coords[cell_conn]
            # Plot cell edges
            if self.dimension == 2:  # For 2D, cells are polygons
                poly = plt.Polygon(
                    nodes[:, :2],
                    closed=True,
                    fill=None,
                    edgecolor="black",
                    linewidth=0.5,
                )
                ax.add_patch(poly)

            if plot_labels:
                # Plot cell centroids
                centroid = self.cell_centroids[i]
                ax.plot(centroid[0], centroid[1], "x", markersize=5, color="red")
                ax.text(centroid[0], centroid[1], f"C{i}", color="red", fontsize=8)

            if plot_element_tags:
                # Display element tags (cell type IDs)
                cell_type_id = self.cell_type_ids[i]
                centroid = self.cell_centroids[i]
                ax.text(
                    centroid[0],
                    centroid[1] + 0.05,
                    f"Tag:{cell_type_id}",
                    color="purple",
                    fontsize=7,
                )

            if plot_areas:
                # Display cell areas
                area = self.cell_volumes[i]  # For 2D, volume is area
                centroid = self.cell_centroids[i]
                ax.text(
                    centroid[0],
                    centroid[1] - 0.05,
                    f"Area:{area:.2f}",
                    color="green",
                    fontsize=7,
                )

            if plot_normals and self._is_analyzed:
                # Plot face normals
                for fi, face_nodes_indices in enumerate(self.cell_faces[i]):
                    face_midpoint = self.face_midpoints[i, fi]
                    face_normal = self.face_normals[i, fi]
                    face_area = self.face_areas[i, fi]

                    # Scale normal for visibility
                    normal_scale = 0.1 * np.max(
                        self.node_coords.max(axis=0) - self.node_coords.min(axis=0)
                    )
                    ax.arrow(
                        face_midpoint[0],
                        face_midpoint[1],
                        face_normal[0] * normal_scale,
                        face_normal[1] * normal_scale,
                        head_width=0.02 * normal_scale,
                        head_length=0.03 * normal_scale,
                        fc="orange",
                        ec="orange",
                    )
                    if (
                        plot_labels
                    ):  # Only label normals if labels are generally enabled
                        ax.text(
                            face_midpoint[0] + face_normal[0] * normal_scale * 1.1,
                            face_midpoint[1] + face_normal[1] * normal_scale * 1.1,
                            f"N{fi}",
                            color="orange",
                            fontsize=6,
                        )
                        ax.text(
                            face_midpoint[0] + face_normal[0] * normal_scale * 1.2,
                            face_midpoint[1] + face_normal[1] * normal_scale * 1.2,
                            f"A:{face_area:.2f}",
                            color="brown",
                            fontsize=6,
                        )

        ax.legend()
        ax.grid(True)
        if show_plot:
            plt.show()

    def _plot_3d_mesh(
        self,
        plot_labels: bool,
        plot_normals: bool,
        plot_element_tags: bool,
        plot_areas: bool,
        show_plot: bool,
    ) -> None:
        """Helper to plot 3D meshes using PyVista."""
        if not pv:
            print("PyVista not available. Cannot plot 3D mesh.")
            return

        # Create a PyVista UnstructuredGrid from the mesh data
        # PyVista expects cell types and connectivity in a specific format
        # This conversion can be complex depending on the variety of cell types
        # For simplicity, let's assume all cells are of a single type for now,
        # or handle common types.
        # A more robust solution would iterate through cell_type_ids and
        # create separate meshes or handle different cell types.

        # PyVista cell types mapping (example for common types)
        # This needs to be comprehensive for all types in self.cell_type_map
        pv_cell_type_map = {
            "tetra": pv.CellType.TETRA,
            "hexahedron": pv.CellType.HEXAHEDRON,
            "triangle": pv.CellType.TRIANGLE,
            "quadrangle": pv.CellType.QUAD,
            # Add more as needed
        }

        # PyVista requires a flat connectivity array with cell sizes
        cells = []
        cell_types = []
        for i, conn in enumerate(self.cell_connectivity):
            cell_type_id = self.cell_type_ids[i]
            cell_info = self.cell_type_map[cell_type_id]
            cell_name = cell_info["name"].lower()  # e.g., 'tetra', 'hexahedron'

            if cell_name in pv_cell_type_map:
                cells.append(len(conn))  # Number of nodes in the cell
                cells.extend(conn)  # Node indices
                cell_types.append(pv_cell_type_map[cell_name])
            else:
                print(
                    f"Warning: Cell type '{cell_name}' not directly supported by PyVista for plotting. Skipping."
                )
                # Fallback: plot as points or edges if type is unknown
                # For now, just skip.

        if not cells:
            print("No supported cells to plot in 3D.")
            return

        grid = pv.UnstructuredGrid(
            np.array(cells), np.array(cell_types), self.node_coords
        )

        plotter = pv.Plotter(window_size=[1024, 768])
        plotter.add_mesh(grid, show_edges=True, color="lightblue", opacity=0.8)

        if plot_labels:
            # Plot cell centroids
            plotter.add_points(
                self.cell_centroids,
                color="red",
                render_points_as_spheres=True,
                point_size=10,
                label="Centroids",
            )
            for i, centroid in enumerate(self.cell_centroids):
                plotter.add_text(f"C{i}", position=centroid, color="red", font_size=8)

        if plot_element_tags:
            for i, centroid in enumerate(self.cell_centroids):
                cell_type_id = self.cell_type_ids[i]
                plotter.add_text(
                    f"Tag:{cell_type_id}",
                    position=centroid + [0, 0, 0.05],
                    color="purple",
                    font_size=7,
                )

        if plot_areas:
            for i, centroid in enumerate(self.cell_centroids):
                area = self.cell_volumes[i]  # For 3D, volume is volume
                plotter.add_text(
                    f"Vol:{area:.2f}",
                    position=centroid + [0, 0, -0.05],
                    color="green",
                    font_size=7,
                )

        if plot_normals and self._is_analyzed:
            # Plot face normals
            for ci in range(self.num_cells):
                for fi in range(len(self.cell_faces[ci])):
                    face_midpoint = self.face_midpoints[ci, fi]
                    face_normal = self.face_normals[ci, fi]
                    face_area = self.face_areas[ci, fi]

                    # Scale normal for visibility
                    # Adjust scale based on mesh size
                    mesh_bounds = grid.bounds
                    max_dim = max(
                        mesh_bounds[1] - mesh_bounds[0],
                        mesh_bounds[3] - mesh_bounds[2],
                        mesh_bounds[5] - mesh_bounds[4],
                    )
                    normal_scale = 0.05 * max_dim

                    plotter.add_mesh(
                        pv.Arrow(face_midpoint, face_normal * normal_scale),
                        color="orange",
                    )
                    if (
                        plot_labels
                    ):  # Only label normals if labels are generally enabled
                        plotter.add_text(
                            f"N{fi}",
                            position=face_midpoint + face_normal * normal_scale * 1.1,
                            color="orange",
                            font_size=6,
                        )
                        plotter.add_text(
                            f"A:{face_area:.2f}",
                            position=face_midpoint + face_normal * normal_scale * 1.2,
                            color="brown",
                            font_size=6,
                        )

        plotter.add_axes()
        plotter.show_grid()
        if show_plot:
            plotter.show()


if __name__ == "__main__":
    print("mesh.py module. Import classes and use in your script.")
