import os
import gmsh


def create_and_mesh_rectangle(
    length,
    height,
    nx,
    ny,
    filename="data/rectangle_mesh.msh",
    mesh_type="structured",
):
    """
    Creates a rectangle, partitions it, and meshes it.

    This function creates a rectangle, adds a horizontal and vertical partition
    through the middle, and then meshes it.
    The mesh_type can be 'structured' (quads), 'triangular', or 'mixed'.
    For 'mixed', the left half is structured and the right half is triangular.

    Args:
        length (float): The length of the rectangle along the x-axis.
        height (float): The height of the rectangle along the y-axis.
        nx (int): The number of elements along the length (x-axis).
        ny (int): The number of elements along the height (y-axis).
        filename (str): The path to save the output .msh file.
        mesh_type (str): Type of mesh to generate.
                         Can be "structured", "triangular", or "mixed".
    """
    if mesh_type not in ["structured", "triangular", "mixed"]:
        raise ValueError("mesh_type must be 'structured', 'triangular', or 'mixed'")

    if mesh_type in ["structured", "mixed"] and (nx % 2 != 0 or ny % 2 != 0):
        print(
            "Warning: For a structured or mixed mesh, it is recommended that nx and ny be even numbers."
        )

    output_dir = os.path.dirname(filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    gmsh.initialize()
    gmsh.model.add(f"{mesh_type}_river")

    # --- Use OpenCASCADE geometry kernel ---
    # Define the corners of the rectangle
    p1 = gmsh.model.occ.addPoint(0, 0, 0)
    p2 = gmsh.model.occ.addPoint(length, 0, 0)
    p3 = gmsh.model.occ.addPoint(length, height, 0)
    p4 = gmsh.model.occ.addPoint(0, height, 0)

    # Define the boundary lines
    l_bottom = gmsh.model.occ.addLine(p1, p2)
    l_right = gmsh.model.occ.addLine(p2, p3)
    l_top = gmsh.model.occ.addLine(p3, p4)
    l_left = gmsh.model.occ.addLine(p4, p1)

    # Create a curve loop and a surface
    cl = gmsh.model.occ.addCurveLoop([l_bottom, l_right, l_top, l_left])
    s = gmsh.model.occ.addPlaneSurface([cl])

    # --- Create partitions ---
    # Create vertical and horizontal lines for partitioning
    l_vertical = gmsh.model.occ.addLine(
        gmsh.model.occ.addPoint(length / 2, 0, 0),
        gmsh.model.occ.addPoint(length / 2, height, 0),
    )
    l_horizontal = gmsh.model.occ.addLine(
        gmsh.model.occ.addPoint(0, height / 2, 0),
        gmsh.model.occ.addPoint(length, height / 2, 0),
    )

    # Use fragment to split the surface with the lines
    gmsh.model.occ.fragment([(2, s)], [(1, l_vertical), (1, l_horizontal)])

    # Synchronize after fragmentation
    gmsh.model.occ.synchronize()

    # --- Meshing ---
    if mesh_type == "structured":
        # For structured mesh, we need to define transfinite properties
        # on all the new curves and surfaces.
        nx_half = nx // 2
        ny_half = ny // 2

        surfaces = gmsh.model.getEntities(2)
        for surface_dim_tag in surfaces:
            surface_tag = surface_dim_tag[1]
            gmsh.model.mesh.setTransfiniteSurface(surface_tag)
            gmsh.model.mesh.setRecombine(2, surface_tag)

            boundary_curves = gmsh.model.getBoundary([surface_dim_tag], oriented=False)
            for curve_dim_tag in boundary_curves:
                curve_tag = curve_dim_tag[1]
                p_tags = gmsh.model.getBoundary([curve_dim_tag], oriented=False)
                p_start_tag = p_tags[0][1]
                p_end_tag = p_tags[1][1]
                coord_start = gmsh.model.occ.getCenterOfMass(0, p_start_tag)
                coord_end = gmsh.model.occ.getCenterOfMass(0, p_end_tag)

                # Check if it's a horizontal or vertical line by checking y-coords
                if abs(coord_start[1] - coord_end[1]) < 1e-6:  # Horizontal
                    gmsh.model.mesh.setTransfiniteCurve(curve_tag, nx_half + 1)
                else:  # Vertical
                    gmsh.model.mesh.setTransfiniteCurve(curve_tag, ny_half + 1)

    elif mesh_type == "triangular":
        # Set a characteristic length for the mesh
        char_length = min(length / nx, height / ny)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", char_length * 0.9)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", char_length * 1.1)

    elif mesh_type == "mixed":
        nx_half = nx // 2
        ny_half = ny // 2

        surfaces = gmsh.model.getEntities(2)
        left_surfaces_tags = []

        # Identify left surfaces
        for surface_dim_tag in surfaces:
            surface_tag = surface_dim_tag[1]
            com = gmsh.model.occ.getCenterOfMass(2, surface_tag)
            if com[0] < length / 2:
                left_surfaces_tags.append(surface_tag)

        # Apply structured mesh properties to left surfaces
        for surface_tag in left_surfaces_tags:
            gmsh.model.mesh.setTransfiniteSurface(surface_tag)
            gmsh.model.mesh.setRecombine(2, surface_tag)

            boundary_curves = gmsh.model.getBoundary([(2, surface_tag)], oriented=False)
            for curve_dim_tag in boundary_curves:
                curve_tag = curve_dim_tag[1]
                p_tags = gmsh.model.getBoundary([curve_dim_tag], oriented=False)
                p_start_tag = p_tags[0][1]
                p_end_tag = p_tags[1][1]
                coord_start = gmsh.model.occ.getCenterOfMass(0, p_start_tag)
                coord_end = gmsh.model.occ.getCenterOfMass(0, p_end_tag)

                if abs(coord_start[1] - coord_end[1]) < 1e-6:  # Horizontal
                    gmsh.model.mesh.setTransfiniteCurve(curve_tag, nx_half + 1)
                else:  # Vertical
                    gmsh.model.mesh.setTransfiniteCurve(curve_tag, ny_half + 1)

        # Set characteristic length for the triangular part (right side)
        char_length = min(length / nx, height / ny)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", char_length * 0.9)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", char_length * 1.1)

    # --- Physical Groups ---
    # We need to find the new boundary lines after fragmentation
    all_curves = gmsh.model.getEntities(1)
    left_bnd, right_bnd, top_bnd, bottom_bnd = [], [], [], []

    for curve in all_curves:
        com = gmsh.model.occ.getCenterOfMass(curve[0], curve[1])
        if abs(com[0] - 0.0) < 1e-6:
            left_bnd.append(curve[1])
        elif abs(com[0] - length) < 1e-6:
            right_bnd.append(curve[1])
        elif abs(com[1] - 0.0) < 1e-6:
            bottom_bnd.append(curve[1])
        elif abs(com[1] - height) < 1e-6:
            top_bnd.append(curve[1])

    gmsh.model.addPhysicalGroup(1, left_bnd, name="left")
    gmsh.model.addPhysicalGroup(1, right_bnd, name="right")
    gmsh.model.addPhysicalGroup(1, bottom_bnd, name="bottom")
    gmsh.model.addPhysicalGroup(1, top_bnd, name="top")

    all_surfaces = [s[1] for s in gmsh.model.getEntities(2)]
    gmsh.model.addPhysicalGroup(2, all_surfaces, name="fluid")

    # Generate the 2D mesh
    gmsh.model.mesh.generate(2)

    # Save the mesh
    gmsh.write(filename)
    print(f"Successfully created {mesh_type} mesh with approx {nx}x{ny} elements.")
    print(f"Mesh saved to: {filename}")

    # Finalize Gmsh
    gmsh.finalize()


if __name__ == "__main__":
    # Define rectangle dimensions and the number of elements
    rect_length = 100.0
    rect_height = 100.0
    num_elements_x = 12
    num_elements_y = 12

    # --- Create Structured Mesh ---
    print(
        f"Creating a structured mesh of size {rect_length}x{rect_height} "
        f"with {num_elements_x}x{num_elements_y} elements."
    )
    create_and_mesh_rectangle(
        rect_length,
        rect_height,
        num_elements_x,
        num_elements_y,
        filename="data/river_structured.msh",
        mesh_type="structured",
    )
    print("\n" + "=" * 40 + "\n")

    # --- Create Triangular Mesh ---
    print(
        f"Creating a triangular mesh of size {rect_length}x{rect_height} "
        f"with approximately {num_elements_x}x{num_elements_y}x2 element resolution."
    )
    # For triangular mesh, nx and ny control the mesh density at the boundaries
    create_and_mesh_rectangle(
        rect_length,
        rect_height,
        num_elements_x,
        num_elements_y,
        filename="data/river_triangular.msh",
        mesh_type="triangular",
    )
    print("\n" + "=" * 40 + "\n")

    # --- Create Mixed Mesh ---
    print(
        f"Creating a mixed mesh of size {rect_length}x{rect_height} "
        f"with a structured left side and triangular right side."
    )
    create_and_mesh_rectangle(
        rect_length,
        rect_height,
        num_elements_x,
        num_elements_y,
        filename="data/river_mixed.msh",
        mesh_type="mixed",
    )

    print("\nScript finished.")
