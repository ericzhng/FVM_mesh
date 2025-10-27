import os
from typing import List, Tuple

import gmsh
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


class Geometry:
    """
    A class to create and manage 2D geometries using gmsh.

    This class provides methods to create various 2D shapes, plot the geometry,
    and compute its bounding box.

    Attributes:
        name (str): The name of the geometry.
    """

    def __init__(self, name: str = ""):
        """
        Initializes the Geometry class.

        Args:
            name (str, optional): The name of the geometry. Defaults to "Default Geometry".
        """
        self.name = name or "Default Geometry"

    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        """
        Computes the bounding box of the entire geometry.

        Returns:
            Tuple[float, float, float, float]: A tuple containing the minimum and maximum
                                               x and y coordinates (min_x, min_y, max_x, max_y).
        """
        gmsh.model.geo.synchronize()
        min_x, min_y, _, max_x, max_y, _ = gmsh.model.getBoundingBox(-1, -1)
        return min_x, min_y, max_x, max_y

    def plot(self, file_path: str = "geometry.png"):
        """
        Plots the wireframe of the current gmsh model and saves it to a file.

        Args:
            file_path (str, optional): The path to save the plot image. Defaults to "geometry.png".
        """
        plt.figure()

        # Generate 1D mesh to get nodes on curves
        gmsh.model.mesh.generate(1)

        for e in gmsh.model.getEntities(1):
            curve_tag = e[1]
            _, node_coords, _ = gmsh.model.mesh.getNodes(
                dim=1, tag=curve_tag, includeBoundary=True
            )

            if len(node_coords) > 0:
                points = np.array(node_coords).reshape(-1, 3)

                # Sort points based on their parametric coordinates if available
                # This helps in plotting the curve in the correct order
                _, _, parametric_coords = gmsh.model.mesh.getNodes(
                    dim=1,
                    tag=curve_tag,
                    includeBoundary=True,
                    returnParametricCoord=True,
                )
                if len(parametric_coords) > 0:
                    # Sort points based on their parametric coordinates
                    sorted_indices = np.argsort(parametric_coords)
                    points = points[sorted_indices]
                plt.plot(points[:, 0], points[:, 1], "k-")

        plt.title(self.name)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.axis("equal")

        output_dir = os.path.dirname(file_path) or "."
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(file_path)
        plt.close()

    def polygon(
        self,
        points: List[Tuple[float, float]],
        convex_hull: bool = False,
        mesh_size: float = 0.1,
    ) -> int:
        """
        Creates a gmsh geometry from a list of 2D points.

        Args:
            points (List[Tuple[float, float]]): A list of (x, y) coordinates for the polygon's vertices.
            convex_hull (bool, optional): If True, computes the convex hull of the points. Defaults to False.
            mesh_size (float, optional): The characteristic length for the mesh. Defaults to 0.1.

        Returns:
            int: The tag of the created surface.
        """
        if len(points) < 3:
            raise ValueError("At least 3 points are required to create a polygon.")

        points_np = np.array(points)
        if convex_hull:
            hull = ConvexHull(points_np)
            processed_points = points_np[hull.vertices]
        else:
            processed_points = points_np

        gmsh_points = [
            gmsh.model.geo.addPoint(p[0], p[1], 0, mesh_size) for p in processed_points
        ]
        lines = [
            gmsh.model.geo.addLine(
                gmsh_points[i], gmsh_points[(i + 1) % len(gmsh_points)]
            )
            for i in range(len(gmsh_points))
        ]

        curve_loop = gmsh.model.geo.addCurveLoop(lines)
        surface = gmsh.model.geo.addPlaneSurface([curve_loop])
        gmsh.model.geo.synchronize()
        return surface

    def rectangle(
        self,
        length: float,
        width: float,
        x: float = 0.0,
        y: float = 0.0,
        mesh_size: float = 0.1,
    ) -> int:
        """
        Creates a rectangular geometry in gmsh.

        Args:
            length (float): The length of the rectangle.
            width (float): The width of the rectangle.
            x (float, optional): The x-coordinate of the bottom-left corner. Defaults to 0.0.
            y (float, optional): The y-coordinate of the bottom-left corner. Defaults to 0.0.
            mesh_size (float, optional): The characteristic length for the mesh. Defaults to 0.1.

        Returns:
            int: The tag of the created surface.
        """
        p1 = gmsh.model.geo.addPoint(x, y, 0, mesh_size)
        p2 = gmsh.model.geo.addPoint(x + length, y, 0, mesh_size)
        p3 = gmsh.model.geo.addPoint(x + length, y + width, 0, mesh_size)
        p4 = gmsh.model.geo.addPoint(x, y + width, 0, mesh_size)

        lines = [
            gmsh.model.geo.addLine(p1, p2),
            gmsh.model.geo.addLine(p2, p3),
            gmsh.model.geo.addLine(p3, p4),
            gmsh.model.geo.addLine(p4, p1),
        ]

        curve_loop = gmsh.model.geo.addCurveLoop(lines)
        surface = gmsh.model.geo.addPlaneSurface([curve_loop])
        gmsh.model.geo.synchronize()
        return surface

    def circle(
        self, radius: float, x: float = 0.0, y: float = 0.0, mesh_size: float = 0.1
    ) -> int:
        """
        Creates a circular geometry in gmsh.

        Args:
            radius (float): The radius of the circle.
            x (float, optional): The x-coordinate of the center. Defaults to 0.0.
            y (float, optional): The y-coordinate of the center. Defaults to 0.0.
            mesh_size (float, optional): The characteristic length for the mesh. Defaults to 0.1.

        Returns:
            int: The tag of the created surface.
        """
        center = gmsh.model.geo.addPoint(x, y, 0, mesh_size)
        p1 = gmsh.model.geo.addPoint(x + radius, y, 0, mesh_size)
        p2 = gmsh.model.geo.addPoint(x, y + radius, 0, mesh_size)
        p3 = gmsh.model.geo.addPoint(x - radius, y, 0, mesh_size)
        p4 = gmsh.model.geo.addPoint(x, y - radius, 0, mesh_size)

        arcs = [
            gmsh.model.geo.addCircleArc(p1, center, p2),
            gmsh.model.geo.addCircleArc(p2, center, p3),
            gmsh.model.geo.addCircleArc(p3, center, p4),
            gmsh.model.geo.addCircleArc(p4, center, p1),
        ]

        curve_loop = gmsh.model.geo.addCurveLoop(arcs)
        surface = gmsh.model.geo.addPlaneSurface([curve_loop])
        gmsh.model.geo.remove([(0, center)], recursive=False)
        gmsh.model.geo.synchronize()
        return surface

    def triangle(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        p3: Tuple[float, float],
        mesh_size: float = 0.1,
    ) -> int:
        """
        Creates a triangular geometry in gmsh.

        Args:
            p1 (Tuple[float, float]): The coordinates of the first vertex.
            p2 (Tuple[float, float]): The coordinates of the second vertex.
            p3 (Tuple[float, float]): The coordinates of the third vertex.
            mesh_size (float, optional): The characteristic length for the mesh. Defaults to 0.1.

        Returns:
            int: The tag of the created surface.
        """
        pt1 = gmsh.model.geo.addPoint(p1[0], p1[1], 0, mesh_size)
        pt2 = gmsh.model.geo.addPoint(p2[0], p2[1], 0, mesh_size)
        pt3 = gmsh.model.geo.addPoint(p3[0], p3[1], 0, mesh_size)

        lines = [
            gmsh.model.geo.addLine(pt1, pt2),
            gmsh.model.geo.addLine(pt2, pt3),
            gmsh.model.geo.addLine(pt3, pt1),
        ]

        curve_loop = gmsh.model.geo.addCurveLoop(lines)
        surface = gmsh.model.geo.addPlaneSurface([curve_loop])
        gmsh.model.geo.synchronize()
        return surface

    def ellipse(
        self,
        r1: float,
        r2: float,
        x: float = 0.0,
        y: float = 0.0,
        mesh_size: float = 0.1,
    ) -> int:
        """
        Creates an elliptical geometry in gmsh.

        Args:
            r1 (float): The radius along the x-axis.
            r2 (float): The radius along the y-axis.
            x (float, optional): The x-coordinate of the center. Defaults to 0.0.
            y (float, optional): The y-coordinate of the center. Defaults to 0.0.
            mesh_size (float, optional): The characteristic length for the mesh. Defaults to 0.1.

        Returns:
            int: The tag of the created surface.
        """
        center = gmsh.model.geo.addPoint(x, y, 0, mesh_size)
        p1 = gmsh.model.geo.addPoint(x + r1, y, 0, mesh_size)
        p2 = gmsh.model.geo.addPoint(x, y + r2, 0, mesh_size)
        p3 = gmsh.model.geo.addPoint(x - r1, y, 0, mesh_size)
        p4 = gmsh.model.geo.addPoint(x, y - r2, 0, mesh_size)

        arcs = [
            gmsh.model.geo.addEllipseArc(p1, center, p1, p2),
            gmsh.model.geo.addEllipseArc(p2, center, p2, p3),
            gmsh.model.geo.addEllipseArc(p3, center, p3, p4),
            gmsh.model.geo.addEllipseArc(p4, center, p4, p1),
        ]

        curve_loop = gmsh.model.geo.addCurveLoop(arcs)
        surface = gmsh.model.geo.addPlaneSurface([curve_loop])
        gmsh.model.geo.remove([(0, center)], recursive=False)
        gmsh.model.geo.synchronize()
        return surface

    def rectangle_with_partitions(
        self,
        length: float,
        width: float,
        x: float = 0.0,
        y: float = 0.0,
        mesh_size: float = 0.1,
    ) -> list:
        """
        Creates a rectangular geometry with partitions using fragmentation.

        This method creates a rectangle and divides it into four smaller rectangles
        by fragmenting it with horizontal and vertical lines.

        Args:
            length (float): The length of the rectangle.
            width (float): The width of the rectangle.
            x (float, optional): The x-coordinate of the bottom-left corner. Defaults to 0.0.
            y (float, optional): The y-coordinate of the bottom-left corner. Defaults to 0.0.
            mesh_size (float, optional): The characteristic length for the mesh. Defaults to 0.1.

        Returns:
            list: A list of tags for the new surfaces created by fragmentation.
        """
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)

        rectangle = gmsh.model.occ.addRectangle(x, y, 0, length, width)

        p_mid_y1 = gmsh.model.occ.addPoint(x, y + width / 2, 0, mesh_size)
        p_mid_y2 = gmsh.model.occ.addPoint(x + length, y + width / 2, 0, mesh_size)
        line_horiz = gmsh.model.occ.addLine(p_mid_y1, p_mid_y2)

        p_mid_x1 = gmsh.model.occ.addPoint(x + length / 2, y, 0, mesh_size)
        p_mid_x2 = gmsh.model.occ.addPoint(x + length / 2, y + width, 0, mesh_size)
        line_vert = gmsh.model.occ.addLine(p_mid_x1, p_mid_x2)

        # Fragment the rectangle with the lines
        fragmented_entities = gmsh.model.occ.fragment(
            [(2, rectangle)], [(1, line_horiz), (1, line_vert)]
        )

        gmsh.model.occ.synchronize()

        surfaces = [s[1] for s in fragmented_entities[0] if s[0] == 2]
        return surfaces
