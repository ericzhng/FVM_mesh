import os

import gmsh
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle
from matplotlib.collections import PatchCollection


class MeshGenerator:
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
            mesh_type = params.get("mesh_type", "tri")

            if mesh_type not in ["structured", "tri", "quads"]:
                raise ValueError("mesh_type must be 'structured', 'tri', or 'quads'")

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
        """Plots the generated mesh with cell and node labels."""
        fig, ax = plt.subplots(figsize=(10, 8))

        # Get nodes
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        x = node_coords[0::3]
        y = node_coords[1::3]
        node_map = {tag: i for i, tag in enumerate(node_tags)}
        num_nodes = len(node_tags)

        # Get cells
        elem_tags_global = []
        elem_node_tags_global = []
        elem_types_global = []
        for surface_tag in self.surface_tags:
            elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2, surface_tag)
            for i, elem_type in enumerate(elem_types):
                elem_types_global.append(elem_type)
                elem_tags_global.append(elem_tags[i])
                elem_node_tags_global.append(elem_node_tags[i])

        num_cells = sum(len(tags) for tags in elem_tags_global)

        # Determine dynamic font size
        base_font_size_cell = 14
        base_font_size_node = 12
        cell_font_scale = max(0.5, 1 - np.log10(num_cells + 1) / 3) if num_cells > 0 else 1
        node_font_scale = max(0.5, 1 - np.log10(num_nodes + 1) / 3) if num_nodes > 0 else 1
        cell_fontsize = base_font_size_cell * cell_font_scale
        node_fontsize = base_font_size_node * node_font_scale

        patches = []
        cell_counter = 0
        for i, elem_type in enumerate(elem_types_global):
            num_nodes_per_elem = 0
            if elem_type == 2:  # 3-node triangle
                num_nodes_per_elem = 3
                color = "#87CEEB"
            elif elem_type == 3:  # 4-node quad
                num_nodes_per_elem = 4
                color = "#90EE90"
            else:
                color = "#FFD700"

            if num_nodes_per_elem > 0:
                num_elem = len(elem_tags_global[i])
                for j in range(num_elem):
                    node_tags_for_elem = elem_node_tags_global[i][j * num_nodes_per_elem : (j + 1) * num_nodes_per_elem]
                    node_indices = [node_map[tag] for tag in node_tags_for_elem]
                    points = np.array([[x[k], y[k]] for k in node_indices])
                    polygon = Polygon(points, facecolor=color, edgecolor="k", alpha=0.6)
                    patches.append(polygon)

                    # Add cell labels
                    cell_centroid_x = float(np.mean(points[:, 0]))
                    cell_centroid_y = float(np.mean(points[:, 1]))
                    ax.text(
                        cell_centroid_x,
                        cell_centroid_y,
                        str(cell_counter),
                        color="black",
                        ha="center",
                        va="center",
                        fontsize=cell_fontsize,
                        weight="bold",
                        bbox=dict(
                            facecolor="white",
                            alpha=0.7,
                            edgecolor="none",
                            boxstyle="round,pad=0.2",
                        ),
                    )
                    cell_counter += 1

        # Add node labels
        for i in range(num_nodes):
            ax.text(
                float(x[i]),
                float(y[i]),
                str(i),
                color="darkred",
                ha="center",
                va="center",
                fontsize=node_fontsize,
                bbox=dict(
                    facecolor="yellow",
                    alpha=0.7,
                    edgecolor="none",
                    boxstyle="round,pad=0.1",
                ),
            )

        p = PatchCollection(patches, match_original=True)
        ax.add_collection(p)

        ax.set_title("Generated Mesh with Labels", fontsize=14, weight="bold")
        ax.set_xlabel("X-coordinate", fontsize=12)
        ax.set_ylabel("Y-coordinate", fontsize=12)
        ax.grid(True, linestyle=":", alpha=0.7)
        ax.set_aspect("equal", adjustable="box")
        ax.autoscale_view()

        for spine in ax.spines.values():
            spine.set_edgecolor("gray")
            spine.set_linewidth(1.5)

        legend_handles = [
            Rectangle((0, 0), 1, 1, color="#87CEEB", label="Triangle"),
            Rectangle((0, 0), 1, 1, color="#90EE90", label="Quadrilateral"),
            Rectangle((0, 0), 1, 1, color="#FFD700", label="Other"),
            Line2D([0], [0], marker="o", color="w", label="Cell Labels", markerfacecolor="black", markersize=base_font_size_cell * 0.8),
            Line2D([0], [0], marker="o", color="w", label="Node Labels", markerfacecolor="darkred", markersize=base_font_size_node * 0.8),
        ]

        ax.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.2),
            fontsize=10,
            frameon=True,
            fancybox=True,
            shadow=True,
            ncol=3,
        )

        plot_file = os.path.join(self.output_dir, file_name)
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Mesh plot saved to: {plot_file}")