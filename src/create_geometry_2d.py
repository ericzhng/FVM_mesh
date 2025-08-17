import os
import gmsh
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


class Geometry2D:
    """A class to create and manage 2D geometries using gmsh."""

    def __init__(self, output_dir: str = "."):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot(self, file_name: str = "geometry.png"):
        """Plots the wireframe of the current gmsh model."""
        plt.figure()

        for e in gmsh.model.getEntities(1): # 1D entities (lines)
            boundary = gmsh.model.getBoundary([e], combined=False)
            points = []
            for p in boundary:
                coords = gmsh.model.getValue(0, p[1], [])
                points.append(coords)
            
            if len(points) == 2:
                plt.plot([points[0][0], points[1][0]], [points[0][1], points[1][1]], 'k-')

        plt.title("Geometry Wireframe")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.axis('equal')

        plot_file = os.path.join(self.output_dir, file_name)
        plt.savefig(plot_file)
        plt.close()

    def polygon(
        self,
        points: List[Tuple[float, float]],
        convex_hull: bool = False,
        mesh_size: float = 0.1,
    ) -> int:
        """Creates a gmsh geometry from a list of 2D points."""
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
        """Creates a rectangular geometry in gmsh."""
        rectangle = gmsh.model.occ.addRectangle(x, y, 0, length, width)
        gmsh.model.occ.synchronize()
        return rectangle

    def circle(
        self, radius: float, x: float = 0.0, y: float = 0.0, mesh_size: float = 0.1
    ) -> int:
        """Creates a circular geometry in gmsh."""
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
        gmsh.model.geo.synchronize()
        return surface

    def triangle(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        p3: Tuple[float, float],
        mesh_size: float = 0.1,
    ) -> int:
        """Creates a triangular geometry in gmsh."""
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
        """Creates an elliptical geometry in gmsh."""
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
        """Creates a rectangular geometry with partitions using fragmentation."""
        # Create the main rectangle
        rectangle = gmsh.model.occ.addRectangle(x, y, 0, length, width)

        # Create partitioning lines (tools)
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

        # Return the new surfaces
        surfaces = [s[1] for s in fragmented_entities[0] if s[0] == 2]
        return surfaces
