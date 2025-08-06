import gmsh
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


class Mesh:
    """
    A class to represent a computational mesh for 1D, 2D, or 3D simulations,
    providing geometric and connectivity information for Finite Volume Methods.
    """

    def __init__(self):
        """
        Initializes the Mesh object with empty attributes.
        """
        self.dim = 0
        self.nelem = 0
        self.nnode = 0

        # raw data
        self.node_tags = np.array([])
        self.elem_tags = np.array([])
        self.node_coords = np.array([])
        self.elem_conn = []  # Use a list to support mixed element types
        self.elem_types = {}

        # derived - boundary
        self.boundary_faces_nodes = np.array([])
        self.boundary_faces_tags = np.array([])
        self.boundary_tag_map = {}

        # derived - variables used in FVM
        self.cell_volumes = np.array([])
        self.cell_centroids = np.array([])
        self.face_normals = np.array([])
        self.face_tangentials = np.array([])
        self.face_areas = np.array([])
        self.cell_neighbors = np.array([])
        self.elem_faces = []

    def read_mesh(self, mesh_file):
        """
        Reads the mesh file using gmsh, determines the highest dimension,
        and extracts node and element information for all element types of that dimension.

        Args:
            mesh_file (str): Path to the mesh file (e.g., .msh).
        """
        gmsh.initialize()
        gmsh.open(mesh_file)

        self.node_tags, self.node_coords, _ = gmsh.model.mesh.getNodes()
        self.node_coords = np.array(self.node_coords).reshape(-1, 3)
        self.nnode = len(self.node_tags)

        elem_types, elem_tags_list, node_connectivity_list = (
            gmsh.model.mesh.getElements()
        )
        max_dim = 0
        for e_type in elem_types:
            _, dim, _, _, _, _ = gmsh.model.mesh.getElementProperties(e_type)
            if dim > max_dim:
                max_dim = dim
        self.dim = max_dim

        all_elem_tags = []
        all_elem_conn = []
        elem_type_counter = 0

        for i, e_type in enumerate(elem_types):
            props = gmsh.model.mesh.getElementProperties(e_type)
            dim = props[1]
            if dim == self.dim:
                num_nodes = props[3]
                name, _, _, _, _, _ = gmsh.model.mesh.getElementProperties(e_type)
                self.elem_types[elem_type_counter] = {
                    "name": name,
                    "num_nodes": num_nodes,
                }

                tags = elem_tags_list[i]
                conn = np.array(node_connectivity_list[i]).reshape(-1, num_nodes)

                all_elem_tags.append(tags)
                all_elem_conn.extend(conn.tolist())
                elem_type_counter += 1

        if all_elem_tags:
            self.elem_tags = np.concatenate(all_elem_tags)
            self.elem_conn = all_elem_conn
            self.nelem = len(self.elem_tags)
        else:
            self.elem_tags = np.array([])
            self.elem_conn = []
            self.nelem = 0

        self._get_boundary_info()
        gmsh.finalize()

    def analyze_mesh(self):
        """
        Analyzes the mesh to compute all geometric and connectivity properties
        required for a Finite Volume Method solver.
        """
        if len(self.node_tags) == 0:
            raise RuntimeError("Mesh data has not been read. Call read_mesh() first.")

        max_tag = np.max(self.node_tags)
        self.node_tag_map = np.full(max_tag + 1, -1, dtype=np.int32)
        self.node_tag_map[self.node_tags] = np.arange(self.nnode, dtype=np.int32)

        self._compute_cell_centroids()
        self._compute_mesh_properties()
        self._compute_cell_volumes()

    def _get_boundary_info(self):
        """
        Extracts boundary faces and their corresponding physical group tags.
        """
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
                b_elem_types, b_elem_tags, b_node_tags = gmsh.model.mesh.getElements(
                    dim, entity
                )
                for i, elem_type in enumerate(b_elem_types):
                    _, _, _, num_nodes, _, _ = gmsh.model.mesh.getElementProperties(
                        elem_type
                    )

                    faces_nodes = np.array(b_node_tags[i]).reshape(-1, num_nodes)
                    faces_nodes.sort(axis=1)
                    all_boundary_faces_nodes.append(faces_nodes)
                    all_boundary_faces_tags.extend([tag] * len(faces_nodes))

        if all_boundary_faces_nodes:
            self.boundary_faces_nodes = np.vstack(all_boundary_faces_nodes)
            self.boundary_faces_tags = np.array(all_boundary_faces_tags)

    def _compute_cell_centroids(self):
        """Computes the centroid of each element."""
        centroids = np.zeros((self.nelem, 3))
        for i, conn in enumerate(self.elem_conn):
            node_indices = self.node_tag_map[conn]
            elem_nodes_coords = self.node_coords[node_indices]
            centroids[i] = np.mean(elem_nodes_coords, axis=0)
        self.cell_centroids = centroids

    def _compute_cell_volumes(self):
        """Computes the volume/area of each element."""
        volumes = np.zeros(self.nelem)
        if self.dim == 1:
            for i, conn in enumerate(self.elem_conn):
                node_indices = self.node_tag_map[conn]
                elem_nodes_coords = self.node_coords[node_indices]
                volumes[i] = np.linalg.norm(
                    elem_nodes_coords[1, :] - elem_nodes_coords[0, :]
                )
        elif self.dim == 2:
            for i, conn in enumerate(self.elem_conn):
                node_indices = self.node_tag_map[conn]
                elem_nodes_coords = self.node_coords[node_indices]
                x = elem_nodes_coords[:, 0]
                y = elem_nodes_coords[:, 1]
                volumes[i] = 0.5 * np.abs(
                    np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
                )
        elif self.dim == 3:
            for i in range(self.nelem):
                volume = 0.0
                for j, face_nodes in enumerate(self.elem_faces[i]):
                    face_midpoint = self.face_midpoints[i, j]
                    face_normal = self.face_normals[i, j]
                    face_area = self.face_areas[i, j]
                    volume += np.dot(face_midpoint, face_normal) * face_area
                volumes[i] = volume / 3.0
        self.cell_volumes = volumes

    def _compute_mesh_properties(self):
        """
        Computes cell neighbors and face properties (normals, tangentials, areas).
        """
        face_definitions_3d = {
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

        self.elem_faces = []
        faces_per_elem_list = []
        max_faces = 0

        for conn in self.elem_conn:
            num_nodes = len(conn)
            face_nodes_def = []
            if self.dim == 2:
                face_nodes_def = [[i, (i + 1) % num_nodes] for i in range(num_nodes)]
            elif self.dim == 3:
                if num_nodes in face_definitions_3d:
                    face_nodes_def = face_definitions_3d[num_nodes]
                else:
                    raise NotImplementedError(
                        f"3D elements with {num_nodes} nodes are not supported."
                    )

            faces_per_elem_list.append(len(face_nodes_def))

            elem_face_nodes = []
            for face_def in face_nodes_def:
                elem_face_nodes.append([conn[i] for i in face_def])
            self.elem_faces.append(elem_face_nodes)

        if faces_per_elem_list:
            max_faces = max(faces_per_elem_list)

        self.cell_neighbors = -np.ones((self.nelem, max_faces), dtype=int)
        self.face_normals = np.zeros((self.nelem, max_faces, 3))
        self.face_tangentials = np.zeros((self.nelem, max_faces, 3))
        self.face_areas = np.zeros((self.nelem, max_faces))
        self.face_midpoints = np.zeros((self.nelem, max_faces, 3))
        self.face_to_cell_distances = np.zeros((self.nelem, max_faces, 2))

        if max_faces == 0:
            return

        face_to_elems = {}
        for i in range(self.nelem):
            for j, face_nodes in enumerate(self.elem_faces[i]):
                sorted_face_nodes = tuple(np.sort(face_nodes))
                face_to_elems.setdefault(sorted_face_nodes, []).append(i)

        for i in range(self.nelem):
            for j, face_nodes in enumerate(self.elem_faces[i]):
                elems = face_to_elems[tuple(np.sort(face_nodes))]
                if len(elems) > 1:
                    self.cell_neighbors[i, j] = elems[0] if elems[1] == i else elems[1]

        for i in range(self.nelem):
            for j, face_nodes in enumerate(self.elem_faces[i]):
                node_indices = self.node_tag_map[face_nodes]
                nodes = self.node_coords[node_indices]
                self.face_midpoints[i, j] = np.mean(nodes, axis=0)

                if self.dim == 2:
                    p1, p2 = nodes[0], nodes[1]
                    delta = p2 - p1
                    length = np.linalg.norm(delta)
                    self.face_areas[i, j] = length
                    if length > 1e-9:
                        self.face_normals[i, j, 0] = delta[1] / length
                        self.face_normals[i, j, 1] = -delta[0] / length
                        self.face_tangentials[i, j, 0] = delta[0] / length
                        self.face_tangentials[i, j, 1] = delta[1] / length
                elif self.dim == 3:
                    if len(nodes) >= 3:
                        if len(nodes) == 3:  # Tri
                            v1 = nodes[1] - nodes[0]
                            v2 = nodes[2] - nodes[0]
                            normal = np.cross(v1, v2)
                        elif len(nodes) == 4:  # Quad
                            v1 = nodes[2] - nodes[0]
                            v2 = nodes[3] - nodes[1]
                            normal = np.cross(v1, v2)
                        else:
                            raise NotImplementedError(
                                f"Face with {len(nodes)} nodes not supported."
                            )

                        area = np.linalg.norm(normal) / 2.0
                        self.face_areas[i, j] = area
                        if area > 1e-9:
                            self.face_normals[i, j] = normal / (2.0 * area)
                            self.face_tangentials[i, j] = (
                                nodes[1] - nodes[0]
                            ) / np.linalg.norm(nodes[1] - nodes[0])

        for i in range(self.nelem):
            for j in range(len(self.elem_faces[i])):
                if (
                    np.dot(
                        self.face_normals[i, j],
                        self.face_midpoints[i, j] - self.cell_centroids[i],
                    )
                    < 0
                ):
                    self.face_normals[i, j] *= -1

        d_i = np.linalg.norm(
            self.face_midpoints - self.cell_centroids[:, np.newaxis, :], axis=2
        )
        valid_neighbors = self.cell_neighbors != -1
        neighbor_indices = np.maximum(self.cell_neighbors, 0)
        neighbor_centroids = self.cell_centroids[neighbor_indices]
        d_j = np.linalg.norm(self.face_midpoints - neighbor_centroids, axis=2)
        self.face_to_cell_distances[..., 0] = d_i
        self.face_to_cell_distances[..., 1] = np.where(valid_neighbors, d_j, 0)

    def get_mesh_quality(self, metric="aspect_ratio"):
        """
        Computes mesh quality for each element.
        """
        if self.nelem == 0:
            return np.array([])
        if self.dim == 1:
            return np.ones(self.nelem)

        quality = np.zeros(self.nelem)
        for i, conn in enumerate(self.elem_conn):
            node_indices = self.node_tag_map[conn]
            elem_nodes_coords = self.node_coords[node_indices]

            if self.dim == 2:
                rolled_nodes = np.roll(elem_nodes_coords, -1, axis=0)
                edge_lengths = np.linalg.norm(elem_nodes_coords - rolled_nodes, axis=1)
                min_edge = np.min(edge_lengths)
                max_edge = np.max(edge_lengths)
                if min_edge > 1e-9:
                    quality[i] = max_edge / min_edge
                else:
                    quality[i] = float("inf")
            elif self.dim == 3:
                edge_lengths = []
                for face_nodes in self.elem_faces[i]:
                    face_node_indices = self.node_tag_map[face_nodes]
                    face_nodes_coords = self.node_coords[face_node_indices]
                    rolled_face_nodes = np.roll(face_nodes_coords, -1, axis=0)
                    edge_vectors = face_nodes_coords - rolled_face_nodes
                    edge_lengths.extend(np.linalg.norm(edge_vectors, axis=1))

                if edge_lengths:
                    min_edge = np.min(edge_lengths)
                    max_edge = np.max(edge_lengths)
                    if min_edge > 1e-9:
                        quality[i] = max_edge / min_edge
                    else:
                        quality[i] = float("inf")
        return quality

    def summary(self):
        """
        Prints a summary of the mesh information.
        """
        print("\n--- Mesh Summary ---")
        print(f"Mesh Dimension: {self.dim}D")
        print(f"Number of Nodes: {self.nnode}")
        print(f"Number of Elements: {self.nelem}")

        if self.nelem > 0:
            elem_type_counts = {}
            for conn in self.elem_conn:
                num_nodes = len(conn)
                elem_type_counts[num_nodes] = elem_type_counts.get(num_nodes, 0) + 1

            type_str = ", ".join(
                [
                    f"{count} x {nnodes}-node"
                    for nnodes, count in elem_type_counts.items()
                ]
            )
            print(f"Element Types: {type_str}")

            quality = self.get_mesh_quality()
            avg_quality = np.mean(quality)
            print(f"Average Mesh Quality (Aspect Ratio): {avg_quality:.4f}")

        num_boundary_sets = len(self.boundary_tag_map)
        print(f"Number of Boundary Face Sets: {num_boundary_sets}")
        if num_boundary_sets > 0:
            for name, tag in self.boundary_tag_map.items():
                count = np.sum(self.boundary_faces_tags == tag)
                print(f"  - Boundary '{name}' (tag {tag}): {count} faces")
        print("--------------------\n")

    def get_mesh_data(self):
        """
        Returns all the computed mesh data in a dictionary.
        """
        return {
            "dimension": self.dim,
            "node_tags": self.node_tags,
            "node_coords": self.node_coords,
            "elem_tags": self.elem_tags,
            "elem_conn": self.elem_conn,
            "cell_volumes": self.cell_volumes,
            "cell_centroids": self.cell_centroids,
            "cell_neighbors": self.cell_neighbors,
            "boundary_faces_nodes": self.boundary_faces_nodes,
            "boundary_faces_tags": self.boundary_faces_tags,
            "boundary_tag_map": self.boundary_tag_map,
            "face_areas": self.face_areas,
            "face_normals": self.face_normals,
            "face_tangentials": self.face_tangentials,
        }


def plot_mesh(mesh: Mesh):
    """
    Visualizes the computational mesh, including element and node labels, and face normals.

    This function is useful for debugging and verifying the mesh structure.

    Args:
        mesh (Mesh): The mesh object to visualize.
    """
    if mesh.nelem == 0:
        raise ValueError("Possibly mesh has not been read. Call read_mesh() first")

    fig, ax = plt.subplots(figsize=(12, 12))

    text_flag = mesh.nelem <= 2000

    max_tag = np.max(mesh.node_tags)
    node_tag_map = np.full(max_tag + 1, -1, dtype=np.int32)
    node_tag_map[mesh.node_tags] = np.arange(mesh.nnode, dtype=np.int32)

    for i, elem_nodes_tags in enumerate(mesh.elem_conn):
        node_indices = [node_tag_map[tag] for tag in elem_nodes_tags]
        nodes = mesh.node_coords[np.array(node_indices)]
        polygon = Polygon(nodes[:, :2], edgecolor="b", facecolor="none", lw=0.5)
        ax.add_patch(polygon)
        if text_flag:
            ax.text(
                mesh.cell_centroids[i, 0],
                mesh.cell_centroids[i, 1],
                f"{i} (A={mesh.cell_volumes[i]:.2f})",
                color="blue",
                fontsize=8,
                ha="center",
            )

    if text_flag:
        for i, coord in enumerate(mesh.node_coords):
            ax.text(
                coord[0],
                coord[1],
                str(mesh.node_tags[i]),
                color="red",
                fontsize=8,
                ha="center",
            )

    if text_flag:
        for i in range(mesh.nelem):
            for j, _ in enumerate(mesh.elem_faces[i]):
                midpoint = mesh.face_midpoints[i, j]
                normal = mesh.face_normals[i, j]
                face_to_cell_distances = mesh.face_to_cell_distances[i, j][0]

                normal_scaled = normal * face_to_cell_distances * 0.5

                ax.quiver(
                    midpoint[0],
                    midpoint[1],
                    normal_scaled[0],
                    normal_scaled[1],
                    angles="xy",
                    scale_units="xy",
                    scale=1,
                    color="green",
                    width=0.003,
                )

    ax.set_aspect("equal", "box")
    ax.set_title("Mesh Visualization")
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.grid(False)
    plt.show(block=True)


if __name__ == "__main__":
    try:
        mesh_file = "./data/river_mixed.msh"

        mesh = Mesh()
        mesh.read_mesh(mesh_file)
        mesh.analyze_mesh()

        mesh.summary()

        mesh_data = mesh.get_mesh_data()
        print("\n--- Mesh Data Export ---")
        print(f"First 5 node coordinates:\n{mesh_data['node_coords'][:5]}")
        print(f"First 5 element connectivities:\n{mesh_data['elem_conn'][:5]}")

    except FileNotFoundError:
        print(f"Error: Mesh file not found at {mesh_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

    if mesh.dim == 2:
        plot_mesh(mesh)
