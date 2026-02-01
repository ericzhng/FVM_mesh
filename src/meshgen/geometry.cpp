#include "geometry.hpp"
#include <gmsh.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fvm {

Geometry::Geometry(const std::string& name)
    : name_(name.empty() ? "Default Geometry" : name) {
}

BoundingBox Geometry::getBoundingBox() const {
    gmsh::model::geo::synchronize();

    double minX, minY, minZ, maxX, maxY, maxZ;
    gmsh::model::getBoundingBox(-1, -1, minX, minY, minZ, maxX, maxY, maxZ);

    return {minX, minY, minZ, maxX, maxY, maxZ};
}

int Geometry::polygon(const std::vector<Point2D>& points,
                      bool convexHull,
                      double meshSize) {
    if (points.size() < 3) {
        throw std::invalid_argument("At least 3 points are required to create a polygon.");
    }

    std::vector<Point2D> processedPoints;

    if (convexHull) {
        auto hullIndices = computeConvexHull(points);
        processedPoints.reserve(hullIndices.size());
        for (auto idx : hullIndices) {
            processedPoints.push_back(points[idx]);
        }
    } else {
        processedPoints = points;
    }

    // Create Gmsh points
    std::vector<int> gmshPoints;
    gmshPoints.reserve(processedPoints.size());
    for (const auto& p : processedPoints) {
        int tag = gmsh::model::geo::addPoint(p[0], p[1], 0.0, meshSize);
        gmshPoints.push_back(tag);
    }

    // Create lines connecting the points
    std::vector<int> lines;
    lines.reserve(gmshPoints.size());
    for (std::size_t i = 0; i < gmshPoints.size(); ++i) {
        std::size_t next = (i + 1) % gmshPoints.size();
        int tag = gmsh::model::geo::addLine(gmshPoints[i], gmshPoints[next]);
        lines.push_back(tag);
    }

    // Create curve loop and surface
    int curveLoop = gmsh::model::geo::addCurveLoop(lines);
    int surface = gmsh::model::geo::addPlaneSurface({curveLoop});

    gmsh::model::geo::synchronize();
    return surface;
}

int Geometry::rectangle(double length, double width,
                        double x, double y,
                        double meshSize) {
    // Create corner points
    int p1 = gmsh::model::geo::addPoint(x, y, 0.0, meshSize);
    int p2 = gmsh::model::geo::addPoint(x + length, y, 0.0, meshSize);
    int p3 = gmsh::model::geo::addPoint(x + length, y + width, 0.0, meshSize);
    int p4 = gmsh::model::geo::addPoint(x, y + width, 0.0, meshSize);

    // Create lines
    std::vector<int> lines = {
        gmsh::model::geo::addLine(p1, p2),  // bottom
        gmsh::model::geo::addLine(p2, p3),  // right
        gmsh::model::geo::addLine(p3, p4),  // top
        gmsh::model::geo::addLine(p4, p1)   // left
    };

    // Create curve loop and surface
    int curveLoop = gmsh::model::geo::addCurveLoop(lines);
    int surface = gmsh::model::geo::addPlaneSurface({curveLoop});

    gmsh::model::geo::synchronize();
    return surface;
}

int Geometry::circle(double radius,
                     double x, double y,
                     double meshSize) {
    // Create center and cardinal points
    int center = gmsh::model::geo::addPoint(x, y, 0.0, meshSize);
    int p1 = gmsh::model::geo::addPoint(x + radius, y, 0.0, meshSize);
    int p2 = gmsh::model::geo::addPoint(x, y + radius, 0.0, meshSize);
    int p3 = gmsh::model::geo::addPoint(x - radius, y, 0.0, meshSize);
    int p4 = gmsh::model::geo::addPoint(x, y - radius, 0.0, meshSize);

    // Create arcs
    std::vector<int> arcs = {
        gmsh::model::geo::addCircleArc(p1, center, p2),
        gmsh::model::geo::addCircleArc(p2, center, p3),
        gmsh::model::geo::addCircleArc(p3, center, p4),
        gmsh::model::geo::addCircleArc(p4, center, p1)
    };

    // Create curve loop and surface
    int curveLoop = gmsh::model::geo::addCurveLoop(arcs);
    int surface = gmsh::model::geo::addPlaneSurface({curveLoop});

    // Remove center point (not needed for meshing)
    gmsh::model::geo::remove({{0, center}}, false);

    gmsh::model::geo::synchronize();
    return surface;
}

int Geometry::triangle(const Point2D& p1, const Point2D& p2, const Point2D& p3,
                       double meshSize) {
    // Create points
    int pt1 = gmsh::model::geo::addPoint(p1[0], p1[1], 0.0, meshSize);
    int pt2 = gmsh::model::geo::addPoint(p2[0], p2[1], 0.0, meshSize);
    int pt3 = gmsh::model::geo::addPoint(p3[0], p3[1], 0.0, meshSize);

    // Create lines
    std::vector<int> lines = {
        gmsh::model::geo::addLine(pt1, pt2),
        gmsh::model::geo::addLine(pt2, pt3),
        gmsh::model::geo::addLine(pt3, pt1)
    };

    // Create curve loop and surface
    int curveLoop = gmsh::model::geo::addCurveLoop(lines);
    int surface = gmsh::model::geo::addPlaneSurface({curveLoop});

    gmsh::model::geo::synchronize();
    return surface;
}

int Geometry::ellipse(double r1, double r2,
                      double x, double y,
                      double meshSize) {
    // Create center and points on ellipse
    int center = gmsh::model::geo::addPoint(x, y, 0.0, meshSize);
    int p1 = gmsh::model::geo::addPoint(x + r1, y, 0.0, meshSize);
    int p2 = gmsh::model::geo::addPoint(x, y + r2, 0.0, meshSize);
    int p3 = gmsh::model::geo::addPoint(x - r1, y, 0.0, meshSize);
    int p4 = gmsh::model::geo::addPoint(x, y - r2, 0.0, meshSize);

    // Create ellipse arcs
    std::vector<int> arcs = {
        gmsh::model::geo::addEllipseArc(p1, center, p1, p2),
        gmsh::model::geo::addEllipseArc(p2, center, p2, p3),
        gmsh::model::geo::addEllipseArc(p3, center, p3, p4),
        gmsh::model::geo::addEllipseArc(p4, center, p4, p1)
    };

    // Create curve loop and surface
    int curveLoop = gmsh::model::geo::addCurveLoop(arcs);
    int surface = gmsh::model::geo::addPlaneSurface({curveLoop});

    // Remove center point
    gmsh::model::geo::remove({{0, center}}, false);

    gmsh::model::geo::synchronize();
    return surface;
}

std::vector<int> Geometry::rectangleWithPartitions(
    double length, double width,
    double x, double y,
    double meshSize) {

    gmsh::option::setNumber("Mesh.CharacteristicLengthMax", meshSize);

    // Create rectangle using OCC kernel
    int rectangle = gmsh::model::occ::addRectangle(x, y, 0.0, length, width);

    // Create horizontal partition line
    int pMidY1 = gmsh::model::occ::addPoint(x, y + width / 2.0, 0.0, meshSize);
    int pMidY2 = gmsh::model::occ::addPoint(x + length, y + width / 2.0, 0.0, meshSize);
    int lineHoriz = gmsh::model::occ::addLine(pMidY1, pMidY2);

    // Create vertical partition line
    int pMidX1 = gmsh::model::occ::addPoint(x + length / 2.0, y, 0.0, meshSize);
    int pMidX2 = gmsh::model::occ::addPoint(x + length / 2.0, y + width, 0.0, meshSize);
    int lineVert = gmsh::model::occ::addLine(pMidX1, pMidX2);

    // Fragment the rectangle with the lines
    std::vector<std::pair<int, int>> objectDimTags = {{2, rectangle}};
    std::vector<std::pair<int, int>> toolDimTags = {{1, lineHoriz}, {1, lineVert}};
    std::vector<std::pair<int, int>> outDimTags;
    std::vector<std::vector<std::pair<int, int>>> outDimTagsMap;

    gmsh::model::occ::fragment(objectDimTags, toolDimTags, outDimTags, outDimTagsMap);
    gmsh::model::occ::synchronize();

    // Extract surface tags
    std::vector<int> surfaces;
    for (const auto& dimTag : outDimTags) {
        if (dimTag.first == 2) {
            surfaces.push_back(dimTag.second);
        }
    }

    return surfaces;
}

std::vector<std::size_t> Geometry::computeConvexHull(const std::vector<Point2D>& points) const {
    // Gift wrapping algorithm for convex hull
    std::size_t n = points.size();
    if (n < 3) {
        std::vector<std::size_t> result(n);
        for (std::size_t i = 0; i < n; ++i) result[i] = i;
        return result;
    }

    // Find the leftmost point
    std::size_t leftmost = 0;
    for (std::size_t i = 1; i < n; ++i) {
        if (points[i][0] < points[leftmost][0]) {
            leftmost = i;
        }
    }

    std::vector<std::size_t> hull;
    std::size_t p = leftmost;

    do {
        hull.push_back(p);
        std::size_t q = (p + 1) % n;

        for (std::size_t i = 0; i < n; ++i) {
            // Cross product to determine orientation
            double cross = (points[i][0] - points[p][0]) * (points[q][1] - points[p][1])
                         - (points[i][1] - points[p][1]) * (points[q][0] - points[p][0]);
            if (cross < 0) {
                q = i;
            }
        }

        p = q;
    } while (p != leftmost);

    return hull;
}

}  // namespace fvm
