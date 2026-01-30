#pragma once

#include "mesh_types.hpp"
#include <string>
#include <tuple>
#include <vector>

namespace fvm {

/**
 * @brief A class to create and manage 2D geometries using Gmsh.
 *
 * This class provides methods to create various 2D shapes (rectangle, circle,
 * polygon, triangle, ellipse) and compute bounding boxes.
 */
class Geometry {
public:
    /**
     * @brief Construct a new Geometry object.
     * @param name The name of the geometry (default: "Default Geometry")
     */
    explicit Geometry(const std::string& name = "");

    /**
     * @brief Get the geometry name.
     * @return The name of the geometry
     */
    const std::string& getName() const { return name_; }

    /**
     * @brief Compute the bounding box of the entire geometry.
     * @return BoundingBox {min_x, min_y, min_z, max_x, max_y, max_z}
     */
    BoundingBox getBoundingBox() const;

    /**
     * @brief Create a polygon from a list of 2D points.
     * @param points List of (x, y) coordinates for vertices
     * @param convexHull If true, compute convex hull of points
     * @param meshSize Characteristic mesh size
     * @return Surface tag of the created polygon
     */
    int polygon(const std::vector<Point2D>& points,
                bool convexHull = false,
                double meshSize = 0.1);

    /**
     * @brief Create a rectangular geometry.
     * @param length Length of the rectangle (x-direction)
     * @param width Width of the rectangle (y-direction)
     * @param x X-coordinate of bottom-left corner
     * @param y Y-coordinate of bottom-left corner
     * @param meshSize Characteristic mesh size
     * @return Surface tag of the created rectangle
     */
    int rectangle(double length, double width,
                  double x = 0.0, double y = 0.0,
                  double meshSize = 0.1);

    /**
     * @brief Create a circular geometry.
     * @param radius Radius of the circle
     * @param x X-coordinate of center
     * @param y Y-coordinate of center
     * @param meshSize Characteristic mesh size
     * @return Surface tag of the created circle
     */
    int circle(double radius,
               double x = 0.0, double y = 0.0,
               double meshSize = 0.1);

    /**
     * @brief Create a triangular geometry.
     * @param p1 First vertex coordinates
     * @param p2 Second vertex coordinates
     * @param p3 Third vertex coordinates
     * @param meshSize Characteristic mesh size
     * @return Surface tag of the created triangle
     */
    int triangle(const Point2D& p1, const Point2D& p2, const Point2D& p3,
                 double meshSize = 0.1);

    /**
     * @brief Create an elliptical geometry.
     * @param r1 Radius along x-axis
     * @param r2 Radius along y-axis
     * @param x X-coordinate of center
     * @param y Y-coordinate of center
     * @param meshSize Characteristic mesh size
     * @return Surface tag of the created ellipse
     */
    int ellipse(double r1, double r2,
                double x = 0.0, double y = 0.0,
                double meshSize = 0.1);

    /**
     * @brief Create a rectangle divided into partitions.
     * @param length Length of the rectangle
     * @param width Width of the rectangle
     * @param x X-coordinate of bottom-left corner
     * @param y Y-coordinate of bottom-left corner
     * @param meshSize Characteristic mesh size
     * @return Vector of surface tags for the created partitions
     */
    std::vector<int> rectangleWithPartitions(
        double length, double width,
        double x = 0.0, double y = 0.0,
        double meshSize = 0.1);

private:
    std::string name_;

    /// Compute convex hull of 2D points (returns indices in CCW order)
    std::vector<std::size_t> computeConvexHull(const std::vector<Point2D>& points) const;
};

}  // namespace fvm
