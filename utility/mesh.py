import os
import gmsh
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from typing import Dict, List, Any, Tuple

os.environ["METIS_DLL"] = os.path.join(os.getcwd(), "dll", "metis.dll")


class Mesh:
    """
    Represents a computational mesh for 1D, 2D, or 3D simulations,
    providing geometric and connectivity information for Finite Volume Methods.
    """

    def __init__(self) -> None:
        """Initializes the Mesh object with empty attributes."""
        self.dim: int = 0
        self.nelem: int = 0
        self.nnode: int = 0

        # Raw data
        self.node_tags: np.ndarray = np.array([])
        self.node_coords: np.ndarray = np.array([])
        self.elem_tags: np.ndarray = np.array([])
        self.elem_conn: List[List[int]] = []
        self.elem_types: Dict[int, Dict[str, Any]] = {}
        self.elem_type_ids: np.ndarray = np.array([])

        # Boundary information
        self.boundary_faces_nodes: np.ndarray = np.array([])
        self.boundary_faces_tags: np.ndarray = np.array([])
        self.boundary_tag_map: Dict[str, int] = {}

        # FVM-related derived data
        self.cell_volumes: np.ndarray = np.array([])
        self.cell_centroids: np.ndarray = np.array([])
        self.face_normals: np.ndarray = np.array([])
        self.face_areas: np.ndarray = np.array([])
        self.cell_neighbors: np.ndarray = np.array([])
        self.elem_faces: List[List[List[int]]] = []
        self.face_midpoints: np.ndarray = np.array([])
        self.face_to_cell_distances: np.ndarray = np.array([])
        self.node_renumber_map: Dict[int, int] = {}
        self.elem_parts: np.ndarray = np.array([])

    def read_mesh(self, mesh_file: str) -> None:
        """
        Reads a mesh file using Gmsh and extracts node and element information.
        """
        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 3)
        try:
            gmsh.open(mesh_file)

            raw_node_tags, rww_node_coords, _ = gmsh.model.mesh.getNodes()
            self.node_coords = np.array(rww_node_coords).reshape(-1, 3)
            self.nnode = len(raw_node_tags)
            self.node_tags = np.arange(self.nnode)

            # Create a map from raw gmsh tags to 0-based indices
            self.node_renumber_map = {tag: i for i, tag in enumerate(raw_node_tags)}

            elem_types, elem_tags_list, connectivity_list = (
                gmsh.model.mesh.getElements()
            )

            self.dim = max(
                gmsh.model.mesh.getElementProperties(e_type)[1] for e_type in elem_types
            )

            all_elem_tags = []
            all_elem_conn = []
            all_elem_type_ids = []
            elem_type_counter = 0

            for i, e_type in enumerate(elem_types):
                props = gmsh.model.mesh.getElementProperties(e_type)
                if props[1] == self.dim:
                    num_nodes = props[3]
                    self.elem_types[elem_type_counter] = {
                        "name": props[0],
                        "num_nodes": num_nodes,
                    }

                    tags = elem_tags_list[i]
                    conn = np.array(connectivity_list[i]).reshape(-1, num_nodes)
                    # Remap connectivity from raw tags to 0-based indices
                    conn = np.vectorize(self.node_renumber_map.get)(conn)

                    all_elem_tags.append(tags)
                    all_elem_conn.extend(conn.tolist())
                    all_elem_type_ids.extend([elem_type_counter] * len(tags))
                    elem_type_counter += 1

            if all_elem_tags:
                self.elem_tags = np.concatenate(all_elem_tags)
                self.elem_conn = all_elem_conn
                self.elem_type_ids = np.array(all_elem_type_ids)
                self.nelem = len(self.elem_tags)

            self._get_boundary_info()

        finally:
            gmsh.finalize()

    def renumber_nodes(self, algorithm: str = "sequential", n_parts: int = 2) -> None:
        """
        Renumbers nodes according to the specified algorithm, updating node coordinates,
        element connectivity, and boundary information accordingly.
        For 'partition' algorithm, it partitions elements (cells) and then renumbers nodes.
        """
        if self.nnode == 0:
            return

        if algorithm == "sequential":
            new_order = np.arange(self.nnode)
        elif algorithm == "reverse":
            new_order = np.arange(self.nnode - 1, -1, -1)
        elif algorithm == "random":
            new_order = np.random.permutation(self.nnode)
        elif algorithm in ("spatial_x", "spatial_y", "spatial_z"):
            axis = {"spatial_x": 0, "spatial_y": 1, "spatial_z": 2}[algorithm]
            new_order = np.argsort(self.node_coords[:, axis])

        elif algorithm == "partition":
            # Partitioning is based on the dual graph of the mesh (elements are vertices,
            # shared faces are edges) for accurate FVM domain decomposition.
            try:
                import metis
            except ImportError:
                raise ImportError(
                    "The 'metis' Python package is required for partition-based renumbering. Install with 'pip install metis'."
                )

            # Ensure cell neighbors are computed to build the cell adjacency graph
            if not self.elem_faces:
                self._extract_faces()
            if self.cell_neighbors.size == 0:
                self._find_cell_neighbors()

            # Build cell adjacency graph
            cell_adj = [[] for _ in range(self.nelem)]
            for i in range(self.nelem):
                for neighbor in self.cell_neighbors[i]:
                    if neighbor != -1:  # -1 indicates a boundary face
                        cell_adj[i].append(neighbor)

            # Partition the cell graph
            (edgecuts, elem_parts) = metis.part_graph(cell_adj, nparts=n_parts)
            self.elem_parts = np.array(elem_parts)

            # Assign partition IDs to nodes. If a node is on a boundary between
            # partitions, it is assigned to the partition with the lowest ID.
            node_to_parts = [set() for _ in range(self.nnode)]
            for elem_idx, conn in enumerate(self.elem_conn):
                part_id = self.elem_parts[elem_idx]
                for node_idx in conn:
                    if node_idx < self.nnode:
                        node_to_parts[node_idx].add(part_id)

            node_partitions = np.zeros(self.nnode, dtype=int)
            for i in range(self.nnode):
                if node_to_parts[i]:
                    node_partitions[i] = min(node_to_parts[i])

            # Sort nodes by partition, then by original index within partition
            partitioned_nodes = [[] for _ in range(n_parts)]
            for idx, part in enumerate(node_partitions):
                if part < n_parts:
                    partitioned_nodes[part].append(idx)
            new_order = np.concatenate(
                [np.array(nodes) for nodes in partitioned_nodes if nodes]
            )

        else:
            raise NotImplementedError(
                f"Renumbering algorithm '{algorithm}' is not implemented."
            )

        # Reorder node coordinates
        self.node_coords = self.node_coords[new_order]

        # Create a map from old indices to new indices
        remap_indices = np.empty_like(new_order)
        remap_indices[new_order] = np.arange(self.nnode)

        # Update element and boundary connectivity
        self.elem_conn = [list(remap_indices[conn]) for conn in self.elem_conn]
        if self.boundary_faces_nodes.size > 0:
            self.boundary_faces_nodes = remap_indices[self.boundary_faces_nodes]

        # Clear dependent data that needs to be recomputed
        self.elem_faces.clear()
        self.cell_neighbors = np.array([])

    def analyze_mesh(self) -> None:
        """
        Computes geometric and connectivity properties for the mesh.
        """
        if self.nelem == 0:
            raise RuntimeError("Mesh is empty. Call read_mesh() first.")

        self._compute_cell_centroids()
        self._extract_faces()
        self._find_cell_neighbors()
        self._compute_face_properties()
        self._compute_cell_volumes()
        self._orient_face_normals()
        self._compute_face_to_cell_distances()

    def _get_boundary_info(self) -> None:
        """Extracts boundary faces and their physical group tags from Gmsh."""
        boundary_dim = self.dim - 1
        if boundary_dim < 0:
            return

        all_boundary_faces_nodes = []
        all_boundary_faces_tags = []

        physical_groups = gmsh.model.getPhysicalGroups(dim=boundary_dim)
        for dim, tag in physical_groups:
            name = gmsh.model.getPhysicalName(dim, tag)
            self.boundary_tag_map[name] = tag
            entities = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
            for entity in entities:
                _, _, b_node_tags = gmsh.model.mesh.getElements(dim, entity)
                if b_node_tags:
                    num_nodes = gmsh.model.mesh.getElementProperties(
                        gmsh.model.mesh.getElements(dim, entity)[0][0]
                    )[3]
                    faces_nodes = np.array(b_node_tags[0]).reshape(-1, num_nodes)
                    all_boundary_faces_nodes.append(faces_nodes)
                    all_boundary_faces_tags.extend([tag] * len(faces_nodes))

        if all_boundary_faces_nodes:
            self.boundary_faces_nodes = np.vstack(all_boundary_faces_nodes)
            self.boundary_faces_tags = np.array(all_boundary_faces_tags)
            self.boundary_faces_nodes = np.vectorize(self.node_renumber_map.get)(
                self.boundary_faces_nodes
            )

    def _compute_cell_centroids(self) -> None:
        """Computes the centroid of each element using vectorized operations."""
        self.cell_centroids = np.array(
            [np.mean(self.node_coords[conn], axis=0) for conn in self.elem_conn]
        )

    def _extract_faces(self) -> None:
        """Extracts faces for each element based on its dimension."""
        face_definitions = {
            4: [[0, 1, 2], [0, 3, 1], [1, 3, 2], [2, 3, 0]],  # Tetrahedron
            8: [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [0, 1, 5, 4],
                [1, 2, 6, 5],
                [2, 3, 7, 6],
                [3, 0, 4, 7],
            ],  # Hexahedron
            6: [
                [0, 1, 2],
                [3, 4, 5],
                [0, 1, 4, 3],
                [1, 2, 5, 4],
                [2, 0, 3, 5],
            ],  # Wedge
        }

        for conn in self.elem_conn:
            num_nodes = len(conn)
            if self.dim == 2:
                face_nodes_def = [[i, (i + 1) % num_nodes] for i in range(num_nodes)]
            elif self.dim == 3:
                if num_nodes not in face_definitions:
                    raise NotImplementedError(
                        f"3D elements with {num_nodes} nodes are not supported yet."
                    )
                face_nodes_def = face_definitions[num_nodes]
            else:
                face_nodes_def = []

            self.elem_faces.append(
                [[conn[i] for i in face_def] for face_def in face_nodes_def]
            )

    def _find_cell_neighbors(self) -> None:
        """Identifies neighboring cells for each face of each element."""
        max_faces = (
            max(len(faces) for faces in self.elem_faces) if self.elem_faces else 0
        )
        self.cell_neighbors = -np.ones((self.nelem, max_faces), dtype=int)
        face_to_elems: Dict[Tuple[int, ...], List[int]] = {}

        for i, faces in enumerate(self.elem_faces):
            for face_nodes in faces:
                sorted_face = tuple(sorted(face_nodes))
                if sorted_face not in face_to_elems:
                    face_to_elems[sorted_face] = []
                face_to_elems[sorted_face].append(i)

        for i, faces in enumerate(self.elem_faces):
            for j, face_nodes in enumerate(faces):
                sorted_face = tuple(sorted(face_nodes))
                elems = face_to_elems[sorted_face]
                if len(elems) > 1:
                    self.cell_neighbors[i, j] = elems[0] if elems[1] == i else elems[1]

    def _compute_face_properties(self) -> None:
        """Computes area and normal vectors for each face."""
        max_faces = self.cell_neighbors.shape[1]
        self.face_areas = np.zeros((self.nelem, max_faces))
        self.face_normals = np.zeros((self.nelem, max_faces, 3))
        self.face_midpoints = np.zeros((self.nelem, max_faces, 3))

        for i, faces in enumerate(self.elem_faces):
            for j, face_nodes in enumerate(faces):
                nodes = self.node_coords[face_nodes]
                self.face_midpoints[i, j] = np.mean(nodes, axis=0)

                if self.dim == 2:
                    p1, p2 = nodes[0], nodes[1]
                    delta = p2 - p1
                    length = np.linalg.norm(delta)
                    self.face_areas[i, j] = length
                    if length > 1e-9:
                        self.face_normals[i, j] = (
                            np.array([delta[1], -delta[0], 0]) / length
                        )
                elif self.dim == 3 and len(nodes) >= 3:
                    # Triangulate the polygon from its centroid to compute area and normal
                    # This is robust for any planar polygon (quads, etc.)
                    centroid = np.mean(nodes, axis=0)
                    area_vec = np.zeros(3)
                    for k in range(len(nodes)):
                        p1 = nodes[k]
                        p2 = nodes[(k + 1) % len(nodes)]
                        area_vec += np.cross(p1 - centroid, p2 - centroid)
                    area_vec /= 2.0

                    area = np.linalg.norm(area_vec)
                    self.face_areas[i, j] = area
                    if area > 1e-9:
                        self.face_normals[i, j] = area_vec / area

    def _orient_face_normals(self) -> None:
        """Ensures all face normals point outwards from the cell."""
        for i in range(self.nelem):
            for j in range(len(self.elem_faces[i])):
                vector_to_face = self.face_midpoints[i, j] - self.cell_centroids[i]
                if np.dot(self.face_normals[i, j], vector_to_face) < 0:
                    self.face_normals[i, j] *= -1

    def _compute_cell_volumes(self) -> None:
        """Computes the volume (or area for 2D) of each element."""
        self.cell_volumes = np.zeros(self.nelem)
        if self.dim == 1:
            for i, conn in enumerate(self.elem_conn):
                p1, p2 = self.node_coords[conn]
                self.cell_volumes[i] = np.linalg.norm(p2 - p1)
        elif self.dim == 2:
            for i, conn in enumerate(self.elem_conn):
                nodes = self.node_coords[conn]
                x, y = nodes[:, 0], nodes[:, 1]
                self.cell_volumes[i] = 0.5 * np.abs(
                    np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))
                )
        elif self.dim == 3:
            # Using the divergence theorem: Volume = (1/3) * sum(face_midpoint . face_normal * face_area)
            volume_contribution = (
                np.einsum("ijk,ijk->ij", self.face_midpoints, self.face_normals)
                * self.face_areas
            )
            self.cell_volumes = np.sum(volume_contribution, axis=1) / 3.0

    def _compute_face_to_cell_distances(self) -> None:
        """Computes distances from face midpoints to cell centroids."""
        max_faces = self.cell_neighbors.shape[1]
        self.face_to_cell_distances = np.zeros((self.nelem, max_faces, 2))

        d_i = np.linalg.norm(
            self.face_midpoints - self.cell_centroids[:, np.newaxis, :], axis=2
        )
        self.face_to_cell_distances[..., 0] = d_i

        valid_neighbors = self.cell_neighbors != -1
        neighbor_indices = np.maximum(self.cell_neighbors, 0)
        neighbor_centroids = self.cell_centroids[neighbor_indices]
        d_j = np.linalg.norm(self.face_midpoints - neighbor_centroids, axis=2)
        self.face_to_cell_distances[..., 1] = np.where(valid_neighbors, d_j, 0)

    def get_mesh_quality(self, metric: str = "aspect_ratio") -> np.ndarray:
        """
        Computes a quality metric for each element.
        Currently supports 'aspect_ratio'.
        """
        if self.nelem == 0 or self.dim == 1:
            return np.ones(self.nelem)

        quality = np.zeros(self.nelem)
        for i, conn in enumerate(self.elem_conn):
            nodes = self.node_coords[conn]
            if self.dim == 2:
                edge_lengths = np.linalg.norm(
                    np.roll(nodes, -1, axis=0) - nodes, axis=1
                )
            elif self.dim == 3:
                edge_lengths = []
                for face in self.elem_faces[i]:
                    face_nodes = self.node_coords[face]
                    edge_lengths.extend(
                        np.linalg.norm(
                            np.roll(face_nodes, -1, axis=0) - face_nodes, axis=1
                        )
                    )

            min_edge, max_edge = np.min(edge_lengths), np.max(edge_lengths)
            quality[i] = max_edge / min_edge if min_edge > 1e-9 else float("inf")

        return quality

    def summary(self) -> None:
        """Prints a summary of the mesh information."""
        print("\n--- Mesh Summary ---")
        print(f"Mesh Dimension: {self.dim}D")
        print(f"Number of Nodes: {self.nnode}")
        print(f"Number of Elements: {self.nelem}")

        if self.nelem > 0:
            print("Element Types:")
            for type_id, type_info in self.elem_types.items():
                count = np.sum(self.elem_type_ids == type_id)
                print(f"  - {type_info['name']}: {count} elements")

            quality = self.get_mesh_quality()
            print(f"Average Mesh Quality (Aspect Ratio): {np.mean(quality):.4f}")

        print(f"Number of Boundary Face Sets: {len(self.boundary_tag_map)}")
        for name, tag in self.boundary_tag_map.items():
            count = np.sum(self.boundary_faces_tags == tag)
            print(f"  - Boundary '{name}' (tag {tag}): {count} faces")
        print("--------------------")

    def get_mesh_data(self) -> Dict[str, Any]:
        """Returns a dictionary containing essential mesh data for an FVM solver."""
        mesh_data = {
            "dimension": self.dim,
            "node_coords": self.node_coords,
            "elem_conn": self.elem_conn,
            "cell_volumes": self.cell_volumes,
            "cell_centroids": self.cell_centroids,
            "face_areas": self.face_areas,
            "face_normals": self.face_normals,
            "cell_neighbors": self.cell_neighbors,
            "boundary_faces_nodes": self.boundary_faces_nodes,
            "boundary_faces_tags": self.boundary_faces_tags,
            "boundary_tag_map": self.boundary_tag_map,
        }
        if self.elem_parts.size > 0:
            mesh_data["elem_parts"] = self.elem_parts
            mesh_data["partition_boundary_info"] = self.get_partition_boundary_info()
        return mesh_data

    def get_partition_boundary_cells(self) -> List[Tuple[int, int]]:
        """
        Identifies pairs of cells that are neighbors across a partition boundary.

        Returns:
            List[Tuple[int, int]]: A list of tuples, where each tuple contains
            the indices of two cells that are neighbors but are in different
            partitions. The first cell in the tuple will always have the lower
            partition ID.
        """
        if self.cell_neighbors.size == 0:
            self._find_cell_neighbors()
        if not hasattr(self, "elem_parts") or self.elem_parts.size == 0:
            raise RuntimeError(
                "Element partitions have not been computed. Run renumber_nodes with 'partition' algorithm first."
            )

        boundary_cell_pairs = []
        processed_pairs = set()
        for cell_idx, neighbors in enumerate(self.cell_neighbors):
            cell_part = self.elem_parts[cell_idx]
            for neighbor_idx in neighbors:
                if neighbor_idx != -1:  # Not a physical boundary
                    neighbor_part = self.elem_parts[neighbor_idx]
                    if cell_part != neighbor_part:
                        # This is a boundary between partitions
                        # Ensure consistent ordering to avoid duplicates
                        if cell_idx < neighbor_idx:
                            pair = (cell_idx, neighbor_idx)
                        else:
                            pair = (neighbor_idx, cell_idx)

                        if pair not in processed_pairs:
                            boundary_cell_pairs.append(pair)
                            processed_pairs.add(pair)

        return boundary_cell_pairs

    def get_partition_boundary_info(self) -> Dict[int, Dict[int, List[int]]]:
        """
        Groups cells on partition boundaries for MPI communication.

        Returns:
            Dict[int, Dict[int, List[int]]]: A dictionary where each key is a
            partition ID, and the value is another dictionary. This inner
            dictionary maps a neighboring partition ID to a list of cell
            indices in the key partition that border the neighbor.
        """
        boundary_cell_pairs = self.get_partition_boundary_cells()

        partition_info: Dict[int, Dict[int, List[int]]] = {}
        if self.elem_parts.size > 0:
            for part_id in range(np.max(self.elem_parts) + 1):
                partition_info[part_id] = {}

        for cell_a, cell_b in boundary_cell_pairs:
            part_a = self.elem_parts[cell_a]
            part_b = self.elem_parts[cell_b]

            if part_b not in partition_info[part_a]:
                partition_info[part_a][part_b] = []
            partition_info[part_a][part_b].append(cell_a)

            if part_a not in partition_info[part_b]:
                partition_info[part_b][part_a] = []
            partition_info[part_b][part_a].append(cell_b)

        return partition_info


def plot_mesh(mesh: Mesh, show_labels: bool = True, n_parts: int = 1) -> None:
    """
    Visualizes the 2D computational mesh, adapting the plot limits to the mesh size.

    Args:
        mesh: The mesh object to visualize.
        show_labels: If True, displays element/node labels and face normals.
        n_parts: If > 1, plots the mesh partitions.
    """
    if mesh.dim != 2:
        print("Plotting is only supported for 2D meshes.")
        return
    if mesh.nelem == 0:
        raise ValueError("Mesh is empty. Call read_mesh() first.")

    fig, ax = plt.subplots(figsize=(14, 12))

    # Determine plot limits from mesh coordinates
    if mesh.nnode > 0:
        x_coords = mesh.node_coords[:, 0]
        y_coords = mesh.node_coords[:, 1]
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        x_range = x_max - x_min
        y_range = y_max - y_min
        padding_x = x_range * 0.02
        padding_y = y_range * 0.02
        ax.set_xlim(x_min - padding_x, x_max + padding_x)
        ax.set_ylim(y_min - padding_y, y_max + padding_y)

    # Use a colormap for different element types or partitions
    if n_parts > 1 and mesh.elem_parts.size > 0:
        cmap = plt.get_cmap("viridis", n_parts)
        # Plot elements with partition colors
        for i, conn in enumerate(mesh.elem_conn):
            nodes = mesh.node_coords[conn]
            # Determine element partition directly from elem_parts
            part_id = mesh.elem_parts[i]
            polygon = Polygon(
                nodes[:, :2],
                edgecolor="black",
                facecolor=cmap(part_id),
                lw=1.0,
                alpha=0.6,
            )
            ax.add_patch(polygon)

        # Create proxy artists for legend
        legend_patches = []
        for i in range(n_parts):
            legend_patches.append(
                Rectangle((0, 0), 1, 1, color=cmap(i), label=f"Partition {i}")
            )
        ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc="upper left")

    else:
        # Original plotting by element type
        num_types = len(mesh.elem_types)
        if num_types > 1:
            cmap = plt.colormaps.get_cmap("viridis")
            colors = cmap(np.linspace(0, 1, num_types))
        else:
            colors = ["blue"]

        for i, conn in enumerate(mesh.elem_conn):
            nodes = mesh.node_coords[conn]
            elem_type_id = mesh.elem_type_ids[i]
            color = colors[elem_type_id] if num_types > 1 else colors[0]

            polygon = Polygon(nodes[:, :2], edgecolor=color, facecolor="none", lw=1.5)
            ax.add_patch(polygon)

    if show_labels:
        for i, coord in enumerate(mesh.node_coords):
            ax.text(coord[0], coord[1], str(i), color="red", fontsize=8, ha="center")
        for i in range(mesh.nelem):
            ax.text(
                mesh.cell_centroids[i, 0],
                mesh.cell_centroids[i, 1],
                f"{i}",
                color="black",
                fontsize=8,
                ha="center",
            )

    ax.set_aspect("equal", "box")
    ax.set_title("Mesh Visualization")
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.grid(False)
    plt.savefig("mesh_plot_renumber_parition.png", dpi=300, bbox_inches="tight")
    plt.show(block=True)


if __name__ == "__main__":
    try:
        # Note: Ensure the mesh file path is correct.
        mesh_file = "./data/river_mixed.msh"
        mesh = Mesh()
        mesh.read_mesh(mesh_file)
        n_parts = 5
        mesh.renumber_nodes(algorithm="partition", n_parts=n_parts)
        mesh.analyze_mesh()
        mesh.summary()

        # Get and print partition information
        if n_parts > 1 and hasattr(mesh, "elem_parts") and mesh.elem_parts.size > 0:
            print("\n--- Partition Information ---")
            counts = np.bincount(mesh.elem_parts)
            for part_id, count in enumerate(counts):
                print(f"  Partition {part_id}: {count} elements")

            min_elem = np.min(counts)
            max_elem = np.max(counts)
            avg_elem = np.mean(counts)
            load_balance = max_elem / avg_elem if avg_elem > 0 else 0.0

            print(f"\n  Min elements per partition: {min_elem}")
            print(f"  Max elements per partition: {max_elem}")
            print(f"  Avg elements per partition: {avg_elem:.2f}")
            print(f"  Load Balance (Max/Avg): {load_balance:.4f}")
            print("-----------------------------")

        # Get and print boundary cell pairs
        boundary_cells = mesh.get_partition_boundary_cells()
        print("\n--- Partition Boundary Cell Pairs ---")
        print(
            f"Found {len(boundary_cells)} pairs of neighboring cells across partitions."
        )
        for pair in boundary_cells:
            print(
                f"  Cell {pair[0]} (Part {mesh.elem_parts[pair[0]]}) <-> Cell {pair[1]} (Part {mesh.elem_parts[pair[1]]})"
            )
        print("-------------------------------------")

        # Get and print partition boundary info
        if n_parts > 1 and hasattr(mesh, "elem_parts") and mesh.elem_parts.size > 0:
            partition_boundary_info = mesh.get_partition_boundary_info()
            print("\n--- Partition Boundary Info ---")
            for part_id, neighbors in partition_boundary_info.items():
                print(f"  Partition {part_id}:")
                for neighbor_part, cells in neighbors.items():
                    print(
                        f"    - Neighbor Partition {neighbor_part}: {len(cells)} cells, e.g. {cells[:5]}"
                    )
            print("-----------------------------")

        mesh_data = mesh.get_mesh_data()
        print("\n--- Mesh Data Export ---")
        for key, value in mesh_data.items():
            if isinstance(value, np.ndarray):
                print(f"- {key}: array with shape {value.shape}")
            elif isinstance(value, list):
                print(f"- {key}: list with {len(value)} items")
            elif isinstance(value, dict):
                print(f"- {key}: dictionary with {len(value)} keys")
            else:
                print(f"- {key}: {value}")
        print("------------------------")

        if mesh.dim == 2:
            # Labels can be disabled for large meshes to improve performance
            plot_mesh(mesh, show_labels=mesh.nelem < 1000, n_parts=n_parts)

    except FileNotFoundError:
        print(f"Error: Mesh file not found at '{mesh_file}'")
    except ImportError as e:
        print(f"An import error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
