import os
import gmsh
import numpy as np
import matplotlib.pyplot as plt


class Mesh2D:
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
        mesh_type="triangular",
        char_length=0.1,
        filename="mesh.msh",
    ):
        """
        Generates the mesh.

        Args:
            mesh_type (str): Type of mesh to generate ('structured' or 'triangular').
            char_length (float): Characteristic length for the mesh.
            filename (str): The path to save the output .msh file.
        """
        if mesh_type not in ["structured", "triangular"]:
            raise ValueError("mesh_type must be 'structured' or 'triangular'")

        if mesh_type == "structured":
            for surface_tag in self.surface_tags:
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

        elif mesh_type == "triangular":
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", char_length * 0.9)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", char_length * 1.1)

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
        print(f"Successfully created {mesh_type} mesh.")
        print(f"Mesh saved to: {msh_file}")

        # Plot the mesh
        self.plot(filename.replace(".msh", ".png"))

    def plot(self, file_name="mesh.png"):
        """Plots the generated mesh."""
        plt.figure()

        # Get nodes
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        x = node_coords[0::3]
        y = node_coords[1::3]

        # Get elements
        elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements()

        for i, elem_type in enumerate(elem_types):
            if elem_type == 2:  # Triangles
                num_elem = len(elem_tags[i])
                for j in range(num_elem):
                    node_tags_for_elem = elem_node_tags[i][j * 3 : (j + 1) * 3]
                    node_indices = [
                        np.where(node_tags == tag)[0][0] for tag in node_tags_for_elem
                    ]

                    tri_points = np.array([[x[k], y[k]] for k in node_indices])
                    plt.plot(
                        np.append(tri_points[:, 0], tri_points[0, 0]),
                        np.append(tri_points[:, 1], tri_points[0, 1]),
                        "b-",
                    )

            elif elem_type == 3:  # Quads
                num_elem = len(elem_tags[i])
                for j in range(num_elem):
                    node_tags_for_elem = elem_node_tags[i][j * 4 : (j + 1) * 4]
                    node_indices = [
                        np.where(node_tags == tag)[0][0] for tag in node_tags_for_elem
                    ]

                    quad_points = np.array([[x[k], y[k]] for k in node_indices])
                    plt.plot(
                        np.append(quad_points[:, 0], quad_points[0, 0]),
                        np.append(quad_points[:, 1], quad_points[0, 1]),
                        "r-",
                    )

        plt.title("Generated Mesh")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.axis("equal")

        plot_file = os.path.join(self.output_dir, file_name)
        plt.savefig(plot_file)
        plt.close()
        print(f"Mesh plot saved to: {plot_file}")
