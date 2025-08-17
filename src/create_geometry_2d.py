import os
import gmsh
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt


class Geometry2D:
    """
    A class to create and manage 2D geometries using gmsh.
    """

    def __init__(self, output_dir="."):
        """
        Initializes the Geometry2D class.

        Args:
            output_dir (str): The directory to save plots and other output files.
        """
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def plot(self, points, file_name="polygon_geometry.png"):
        """
        Plots the convex hull of a set of points and saves the plot to a file.

        Args:
            points (list of tuples): A list of (x, y) coordinates.
            file_name (str): The name of the output plot file.
        """
        points = np.array(points)
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]

        plt.figure()
        plt.plot(points[:, 0], points[:, 1], "o", label="Original Points")

        # Plot the convex hull by connecting the vertices
        for i in range(len(hull.vertices)):
            p1 = points[hull.vertices[i]]
            p2 = points[hull.vertices[(i + 1) % len(hull.vertices)]]
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], "k-")

        # Fill the convex hull
        plt.fill(
            hull_points[:, 0], hull_points[:, 1], "r", alpha=0.3, label="Convex Hull"
        )

        plt.title("Polygon Geometry from Unorganized Points")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.grid(True)

        # Save the plot
        plot_file = os.path.join(self.output_dir, file_name)
        plt.savefig(plot_file)
        print(f"Plot saved to {plot_file}")
        plt.show()  # Uncomment to display the plot

    def polygon(self, points, convex_hull=False, mesh_size=0.1):
        """
        Creates a gmsh geometry from a list of unorganized 2D points.
        The geometry is based on the convex hull of the points.
        Assumes gmsh is initialized.

        Args:
            points (list of tuples): A list of (x, y) coordinates.
            mesh_size (float): The desired mesh size.

        Returns:
            int: The tag of the created surface.
        """
        if len(points) < 3:
            raise ValueError("At least 3 points are required to create a polygon.")

        points = np.array(points)
        if convex_hull:
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
        else:
            hull_points = points

        # Add points to gmsh
        gmsh_points = []
        for point in hull_points:
            gmsh_points.append(
                gmsh.model.geo.addPoint(point[0], point[1], 0, mesh_size)
            )

        # Create lines
        lines = []
        for i in range(len(gmsh_points)):
            lines.append(
                gmsh.model.geo.addLine(
                    gmsh_points[i], gmsh_points[(i + 1) % len(gmsh_points)]
                )
            )

        # Create curve loop and surface
        curve_loop = gmsh.model.geo.addCurveLoop(lines)
        surface = gmsh.model.geo.addPlaneSurface([curve_loop])

        gmsh.model.geo.synchronize()

        return surface

    def rectangle(self, length, width, x=0.0, y=0.0, mesh_size=0.1):
        """
        Creates a rectangular geometry in gmsh.

        Args:
            x (float): The x-coordinate of the bottom-left corner.
            y (float): The y-coordinate of the bottom-left corner.
            length (float): The length of the rectangle.
            width (float): The width of the rectangle.
            mesh_size (float): The desired mesh size.

        Returns:
            int: The tag of the created surface.
        """
        # Add points to gmsh
        p1 = gmsh.model.geo.addPoint(x, y, 0, mesh_size)
        p2 = gmsh.model.geo.addPoint(x + length, y, 0, mesh_size)
        p3 = gmsh.model.geo.addPoint(x + length, y + width, 0, mesh_size)
        p4 = gmsh.model.geo.addPoint(x, y + width, 0, mesh_size)

        # Create lines
        l1 = gmsh.model.geo.addLine(p1, p2)
        l2 = gmsh.model.geo.addLine(p2, p3)
        l3 = gmsh.model.geo.addLine(p3, p4)
        l4 = gmsh.model.geo.addLine(p4, p1)

        # Create curve loop and surface
        curve_loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
        surface = gmsh.model.geo.addPlaneSurface([curve_loop])

        gmsh.model.geo.synchronize()

        return surface

    def circle(self, radius, x=0.0, y=0.0, mesh_size=0.1):
        """
        Creates a circular geometry in gmsh.

        Args:
            x (float): The x-coordinate of the center.
            y (float): The y-coordinate of the center.
            radius (float): The radius of the circle.
            mesh_size (float): The desired mesh size.

        Returns:
            int: The tag of the created surface.
        """
        # Add center point
        center = gmsh.model.geo.addPoint(x, y, 0, mesh_size)

        # Add points on the circle
        p1 = gmsh.model.geo.addPoint(x + radius, y, 0, mesh_size)
        p2 = gmsh.model.geo.addPoint(x, y + radius, 0, mesh_size)
        p3 = gmsh.model.geo.addPoint(x - radius, y, 0, mesh_size)
        p4 = gmsh.model.geo.addPoint(x, y - radius, 0, mesh_size)

        # Create arcs
        a1 = gmsh.model.geo.addCircleArc(p1, center, p2)
        a2 = gmsh.model.geo.addCircleArc(p2, center, p3)
        a3 = gmsh.model.geo.addCircleArc(p3, center, p4)
        a4 = gmsh.model.geo.addCircleArc(p4, center, p1)

        # Create curve loop and surface
        curve_loop = gmsh.model.geo.addCurveLoop([a1, a2, a3, a4])
        surface = gmsh.model.geo.addPlaneSurface([curve_loop])

        gmsh.model.geo.synchronize()

        return surface

    def triangle(self, p1, p2, p3, mesh_size=0.1):
        """
        Creates a triangular geometry in gmsh.

        Args:
            p1 (tuple): (x, y) coordinates of the first point.
            p2 (tuple): (x, y) coordinates of the second point.
            p3 (tuple): (x, y) coordinates of the third point.
            mesh_size (float): The desired mesh size.

        Returns:
            int: The tag of the created surface.
        """
        # Add points to gmsh
        pt1 = gmsh.model.geo.addPoint(p1[0], p1[1], 0, mesh_size)
        pt2 = gmsh.model.geo.addPoint(p2[0], p2[1], 0, mesh_size)
        pt3 = gmsh.model.geo.addPoint(p3[0], p3[1], 0, mesh_size)

        # Create lines
        l1 = gmsh.model.geo.addLine(pt1, pt2)
        l2 = gmsh.model.geo.addLine(pt2, pt3)
        l3 = gmsh.model.geo.addLine(pt3, pt1)

        # Create curve loop and surface
        curve_loop = gmsh.model.geo.addCurveLoop([l1, l2, l3])
        surface = gmsh.model.geo.addPlaneSurface([curve_loop])

        gmsh.model.geo.synchronize()

        return surface

    def ellipse(self, r1, r2, x=0.0, y=0.0, mesh_size=0.1):
        """
        Creates an elliptical geometry in gmsh.

        Args:
            x (float): The x-coordinate of the center.
            y (float): The y-coordinate of the center.
            r1 (float): The radius along the x-axis.
            r2 (float): The radius along the y-axis.
            mesh_size (float): The desired mesh size.

        Returns:
            int: The tag of the created surface.
        """
        # Add center point
        center = gmsh.model.geo.addPoint(x, y, 0, mesh_size)

        # Add points on the ellipse
        p1 = gmsh.model.geo.addPoint(x + r1, y, 0, mesh_size)
        p2 = gmsh.model.geo.addPoint(x, y + r2, 0, mesh_size)
        p3 = gmsh.model.geo.addPoint(x - r1, y, 0, mesh_size)
        p4 = gmsh.model.geo.addPoint(x, y - r2, 0, mesh_size)

        # Create arcs
        a1 = gmsh.model.geo.addEllipseArc(p1, center, p1, p2)
        a2 = gmsh.model.geo.addEllipseArc(p2, center, p2, p3)
        a3 = gmsh.model.geo.addEllipseArc(p3, center, p3, p4)
        a4 = gmsh.model.geo.addEllipseArc(p4, center, p4, p1)

        # Create curve loop and surface
        curve_loop = gmsh.model.geo.addCurveLoop([a1, a2, a3, a4])
        surface = gmsh.model.geo.addPlaneSurface([curve_loop])

        gmsh.model.geo.synchronize()

        return surface
