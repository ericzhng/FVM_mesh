import os
from typing import List, Dict, Any

import gmsh
import numpy as np
import matplotlib.pyplot as plt

from ..common.utility import plot_mesh


class MeshGenerator:
    """
    A class to generate 2D meshes using gmsh.

    This class handles the generation of structured, triangular, and quadrilateral
    meshes, and provides methods for plotting and saving the generated mesh.

    Attributes:
        surface_tags (List[int]): A list of surface tags to be meshed.
        output_dir (str): The directory to save output files.
    """

    def __init__(self, surface_tags, output_dir: str = "."):
        """
        Initializes the MeshGenerator class.

        Args:
            surface_tags (list or int): A list of surface tags or a single surface tag.
            output_dir (str, optional): The directory to save output files. Defaults to ".".
        """
        self.surface_tags = (
            [surface_tags] if isinstance(surface_tags, int) else surface_tags
        )
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate(
        self,
        mesh_params: Dict[int, Dict[str, Any]],
        filename: str = "mesh.msh",
        show_nodes: bool = False,
        show_cells: bool = False,
    ):
        """
        Generates the mesh for the specified surfaces.

        Args:
            mesh_params (Dict[int, Dict[str, Any]]): A dictionary where keys are surface tags
                and values are dicts with 'mesh_type' and 'char_length'.
            filename (str, optional): The path to save the output .msh file. Defaults to "mesh.msh".
            show_nodes (bool, optional): Whether to show node labels in the plot. Defaults to False.
            show_cells (bool, optional): Whether to show cell labels in the plot. Defaults to False.
        """
        for surface_tag in self.surface_tags:
            if surface_tag in mesh_params:
                self._apply_mesh_parameters(surface_tag, mesh_params[surface_tag])

        self._setup_physical_groups()
        gmsh.model.mesh.generate(2)
        self._save_mesh(filename)
        self.plot(
            file_name=filename.replace(".msh", ".png"),
            show_nodes=show_nodes,
            show_cells=show_cells,
        )

    def _apply_mesh_parameters(self, surface_tag: int, params: Dict[str, Any]):
        """Applies mesh parameters to a given surface."""
        mesh_type = params.get("mesh_type", "tri")
        if mesh_type not in ["structured", "tri", "quads"]:
            raise ValueError("mesh_type must be 'structured', 'tri', or 'quads'")

        if mesh_type == "structured":
            self._set_structured_mesh(surface_tag, params.get("char_length", 0.1))
        elif mesh_type == "quads":
            gmsh.model.mesh.setRecombine(2, surface_tag)

    def _set_structured_mesh(self, surface_tag: int, char_length: float):
        """Sets a structured mesh on a surface."""
        gmsh.model.mesh.setTransfiniteSurface(surface_tag)
        gmsh.model.mesh.setRecombine(2, surface_tag)

        boundary_curves = gmsh.model.getBoundary([(2, surface_tag)], oriented=False)
        if len(boundary_curves) != 4:
            raise ValueError(
                "Structured mesh is only supported for geometries with 4 boundary curves."
            )

        bbox = gmsh.model.getBoundingBox(2, surface_tag)
        dx = bbox[3] - bbox[0]
        dy = bbox[4] - bbox[1]
        nx = int(dx / char_length)
        ny = int(dy / char_length)

        h_curves, v_curves = self._classify_boundary_curves(boundary_curves)

        for curve_tag in h_curves:
            gmsh.model.mesh.setTransfiniteCurve(curve_tag, nx + 1)
        for curve_tag in v_curves:
            gmsh.model.mesh.setTransfiniteCurve(curve_tag, ny + 1)

    def _classify_boundary_curves(self, boundary_curves: list) -> tuple[list, list]:
        """Classifies boundary curves into horizontal and vertical lists."""
        h_curves, v_curves = [], []
        for dim_tag in boundary_curves:
            curve_tag = dim_tag[1]
            p_tags = gmsh.model.getBoundary([dim_tag], oriented=False)
            p_start_tag, p_end_tag = p_tags[0][1], p_tags[1][1]
            coord_start = gmsh.model.occ.getCenterOfMass(0, p_start_tag)
            coord_end = gmsh.model.occ.getCenterOfMass(0, p_end_tag)

            if abs(coord_start[1] - coord_end[1]) < 1e-6:
                h_curves.append(curve_tag)
            else:
                v_curves.append(curve_tag)
        return h_curves, v_curves

    def _setup_physical_groups(self):
        """Sets up physical groups for boundaries and surfaces."""
        all_boundary_curves = []
        for surface_tag in self.surface_tags:
            boundary_curves = gmsh.model.getBoundary([(2, surface_tag)], oriented=False)
            all_boundary_curves.extend([c[1] for c in boundary_curves])

        if all_boundary_curves:
            gmsh.model.addPhysicalGroup(
                1, list(set(all_boundary_curves)), name="boundary"
            )
        if self.surface_tags:
            gmsh.model.addPhysicalGroup(2, self.surface_tags, name="fluid")

    def _save_mesh(self, filename: str):
        """Saves the generated mesh to a file."""
        msh_file = os.path.join(self.output_dir, filename)
        gmsh.write(msh_file)
        print(f"Successfully created mesh and saved to: {msh_file}")

    def plot(
        self,
        file_name: str = "mesh.png",
        show_nodes: bool = False,
        show_cells: bool = False,
    ):
        """
        Plots the generated mesh and saves it to a file.

        Args:
            file_name (str, optional): The name of the output plot file. Defaults to "mesh.png".
            show_nodes (bool, optional): Whether to show node labels. Defaults to False.
            show_cells (bool, optional): Whether to show cell labels. Defaults to False.
        """
        fig, ax = plt.subplots(figsize=(12, 10))

        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        nodes = np.array(node_coords).reshape(-1, 3)
        node_map = {tag: i for i, tag in enumerate(node_tags)}

        cells = self._get_cells(node_map)

        model_name = gmsh.model.getCurrent() or "default"
        plot_mesh(ax, nodes, cells, show_nodes, show_cells, title=f"{model_name}")

        plot_file = os.path.join(self.output_dir, file_name)
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Mesh plot saved to: {plot_file}")

    def _get_cells(self, node_map: Dict[int, int]) -> List[List[int]]:
        """Extracts cell connectivity from the gmsh model."""
        cells = []
        for surface_tag in self.surface_tags:
            elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(
                2, surface_tag
            )
            for i, elem_type in enumerate(elem_types):
                props = gmsh.model.mesh.getElementProperties(elem_type)
                num_nodes = props[3]
                num_elem = len(elem_tags[i])
                elem_nodes = np.array(elem_node_tags[i]).reshape(num_elem, num_nodes)
                cells.extend(
                    [[node_map[tag] for tag in enodes] for enodes in elem_nodes]
                )
        return cells
