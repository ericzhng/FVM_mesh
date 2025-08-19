import os
import gmsh
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


class MeshGen2D:
    """A class to generate 2D meshes using gmsh."""

    def __init__(self, surface_tags, output_dir="."):
        """
        Initializes the Mesh2D class.

        Args:
            surface_tags (list or int): A list of surface tags or a single surface tag.
            output_dir (str): The directory to save the output files.
        """
        if isinstance(surface_tags, int):
            self.surface_tags = [surface_tags]
        else:
            self.surface_tags = surface_tags
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate(
        self,
        mesh_params: dict,
        filename="mesh.msh",
    ):
        """
        Generates the mesh.

        Args:
            mesh_params (dict): A dictionary where keys are surface tags and values are dicts
                                with 'mesh_type' and 'char_length'.
            filename (str): The path to save the output .msh file.
        """
        for surface_tag in self.surface_tags:
            if surface_tag not in mesh_params:
                continue

            params = mesh_params[surface_tag]
            mesh_type = params.get("mesh_type", "triangular")

            if mesh_type not in ["structured", "triangular", "quads"]:
                raise ValueError(
                    "mesh_type must be 'structured', 'triangular', or 'quads'"
                )

            if mesh_type == "structured":
                char_length = params.get("char_length", 0.1)
                # For structured mesh, we need to define transfinite properties
                # on all the new curves and surfaces.
                gmsh.model.mesh.setTransfiniteSurface(surface_tag)
                gmsh.model.mesh.setRecombine(2, surface_tag)

                boundary_curves = gmsh.model.getBoundary(
                    [(2, surface_tag)], oriented=False
                )

                if len(boundary_curves) != 4:
                    raise ValueError(
                        "Structured mesh is only supported for geometries with 4 boundary curves."
                    )

                # Determine nx and ny from char_length
                bbox = gmsh.model.getBoundingBox(2, surface_tag)
                dx = bbox[3] - bbox[0]
                dy = bbox[4] - bbox[1]
                nx = int(dx / char_length)
                ny = int(dy / char_length)

                horizontal_curves = []
                vertical_curves = []

                for curve_dim_tag in boundary_curves:
                    curve_tag = curve_dim_tag[1]
                    p_tags = gmsh.model.getBoundary([curve_dim_tag], oriented=False)
                    p_start_tag = p_tags[0][1]
                    p_end_tag = p_tags[1][1]
                    coord_start = gmsh.model.occ.getCenterOfMass(0, p_start_tag)
                    coord_end = gmsh.model.occ.getCenterOfMass(0, p_end_tag)

                    if abs(coord_start[1] - coord_end[1]) < 1e-6:  # Horizontal
                        horizontal_curves.append(curve_tag)
                    else:  # Vertical
                        vertical_curves.append(curve_tag)

                for curve_tag in horizontal_curves:
                    gmsh.model.mesh.setTransfiniteCurve(curve_tag, nx + 1)

                for curve_tag in vertical_curves:
                    gmsh.model.mesh.setTransfiniteCurve(curve_tag, ny + 1)

            elif mesh_type == "quads":
                gmsh.model.mesh.setRecombine(2, surface_tag)

        # --- Physical Groups ---
        all_boundary_curves = []
        for surface_tag in self.surface_tags:
            boundary_curves = gmsh.model.getBoundary([(2, surface_tag)], oriented=False)
            all_boundary_curves.extend([c[1] for c in boundary_curves])

        gmsh.model.addPhysicalGroup(1, list(set(all_boundary_curves)), name="boundary")
        gmsh.model.addPhysicalGroup(2, self.surface_tags, name="fluid")

        # Generate the 2D mesh
        gmsh.model.mesh.generate(2)

        # Save the mesh
        msh_file = os.path.join(self.output_dir, filename)
        gmsh.write(msh_file)
        print(f"Successfully created mesh.")
        print(f"Mesh saved to: {msh_file}")

        # Plot the mesh
        self.plot(mesh_params, filename.replace(".msh", ".png"))

    def plot(self, mesh_params, file_name="mesh.png"):
        """Plots the generated mesh."""
        fig, ax = plt.subplots(figsize=(10, 8))

        # Get nodes
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        x = node_coords[0::3]
        y = node_coords[1::3]
        node_map = {tag: i for i, tag in enumerate(node_tags)}

        patches = []
        for surface_tag in self.surface_tags:
            mesh_type = mesh_params.get(surface_tag, {}).get("mesh_type", "triangular")

            elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(
                2, surface_tag
            )

            for i, elem_type in enumerate(elem_types):
                if elem_type == 2:  # Triangles
                    color = "blue"
                    if mesh_type in ["structured", "quads"]:
                        color = "yellow"  # Unexpected triangle

                    num_elem = len(elem_tags[i])
                    for j in range(num_elem):
                        node_tags_for_elem = elem_node_tags[i][j * 3 : (j + 1) * 3]
                        node_indices = [node_map[tag] for tag in node_tags_for_elem]
                        tri_points = np.array([[x[k], y[k]] for k in node_indices])
                        polygon = Polygon(tri_points, facecolor=color, edgecolor="k")
                        patches.append(polygon)

                elif elem_type == 3:  # Quads
                    color = "red"
                    if mesh_type == "triangular":
                        color = "green"  # Unexpected quad

                    num_elem = len(elem_tags[i])
                    for j in range(num_elem):
                        node_tags_for_elem = elem_node_tags[i][j * 4 : (j + 1) * 4]
                        node_indices = [node_map[tag] for tag in node_tags_for_elem]
                        quad_points = np.array([[x[k], y[k]] for k in node_indices])
                        polygon = Polygon(quad_points, facecolor=color, edgecolor="k")
                        patches.append(polygon)

        p = PatchCollection(patches, match_original=True)
        ax.add_collection(p)

        plt.title("Generated Mesh")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(False)
        plt.axis("equal")
        ax.autoscale_view()

        plot_file = os.path.join(self.output_dir, file_name)
        plt.savefig(plot_file, dpi=300)
        plt.close()
        print(f"Mesh plot saved to: {plot_file}")
