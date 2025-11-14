from src.fvm_mesh.meshgen.geometry import Geometry
from src.fvm_mesh.meshgen.mesh_generator import MeshGenerator

import gmsh


def main():
    """Create a rectangular mesh and save it to a Gmsh file."""
    # Define geometry parameters
    length = 1.0  # Length of the rectangle in the x-direction
    height = 1.0  # Height of the rectangle in the y-direction

    # Create geometry
    gmsh.initialize()
    gmsh.model.add("test_polygon")

    geom = Geometry()
    surface_tag = geom.rectangle(length, height, mesh_size=0.05)

    # synchronize the geo kernel to the model so model-level API can access entities
    gmsh.model.geo.synchronize()

    # Get the boundary entities of the surface using the model-level API
    boundary_entities = gmsh.model.getBoundary([(2, surface_tag)])

    # Identify the boundary lines by their position
    line_tags = [abs(tag) for dim, tag in boundary_entities]

    top_bc = -1
    bottom_bc = -1
    left_bc = -1
    right_bc = -1

    tol = 1e-9
    for line_tag in line_tags:
        bbox = gmsh.model.getBoundingBox(1, line_tag)
        if abs(bbox[1] - 0.0) < tol and abs(bbox[4] - 0.0) < tol:
            bottom_bc = line_tag
        elif abs(bbox[0] - length) < tol and abs(bbox[3] - length) < tol:
            right_bc = line_tag
        elif abs(bbox[1] - height) < tol and abs(bbox[4] - height) < tol:
            top_bc = line_tag
        elif abs(bbox[0] - 0.0) < tol and abs(bbox[3] - 0.0) < tol:
            left_bc = line_tag

    # Add physical groups for the boundaries
    if left_bc != -1:
        gmsh.model.addPhysicalGroup(1, [left_bc], name="inlet")
    if right_bc != -1:
        gmsh.model.addPhysicalGroup(1, [right_bc], name="outlet")
    if bottom_bc != -1 and top_bc != -1:
        gmsh.model.addPhysicalGroup(1, [bottom_bc, top_bc], name="wall")

    # gmsh.model.addPhysicalGroup(1, line_tags, name="boundary")

    # Generate mesh
    output_dir = "data"
    mesher = MeshGenerator(surface_tags=surface_tag, output_dir=output_dir)
    mesh_filename = "sample_rect_mesh.msh"
    mesh_params = {surface_tag: {"mesh_type": "quads", "char_length": 0.01}}
    mesher.generate(
        mesh_params=mesh_params,
        filename=mesh_filename,
        show_nodes=True,
        show_cells=True,
    )

    gmsh.fltk.run()
    gmsh.finalize()


if __name__ == "__main__":
    main()
