#pragma once

/**
 * @file mesh_quality.hpp
 * @brief Mesh quality metrics computation for PolyMesh objects.
 *
 * This module provides the MeshQuality class for computing and storing
 * various quality metrics that assess the suitability of a mesh for
 * numerical simulations.
 */

#include "mesh_processing_export.hpp"
#include <string>
#include <vector>

namespace fvm {

// Forward declaration
class PolyMesh;

// Constants for geometric computations
constexpr double GEOMETRY_TOLERANCE = 1e-12;
constexpr double IDEAL_TRIANGLE_ANGLE = 60.0;
constexpr double IDEAL_QUAD_ANGLE = 90.0;

/**
 * @brief Stores and computes mesh quality metrics.
 *
 * This class holds the results of a mesh quality analysis, including
 * geometric metrics like skewness, aspect ratio, and non-orthogonality,
 * as well as topological connectivity checks.
 */
class MESH_PROCESSING_API MeshQuality {
public:
    // =========================================================================
    // Quality Metrics Data
    // =========================================================================

    /// Ratio of the smallest to the largest cell volume (0 to 1)
    double minMaxVolumeRatio = 0.0;

    /// Skewness values for each cell (0 = perfect, 1 = degenerate)
    std::vector<double> cellSkewness;

    /// Non-orthogonality values for each cell (in degrees)
    std::vector<double> cellNonOrthogonality;

    /// Aspect ratio values for each cell (>= 1, lower is better)
    std::vector<double> cellAspectRatio;

    /// List of connectivity issues found in the mesh
    std::vector<std::string> connectivityIssues;

    // =========================================================================
    // Factory Method
    // =========================================================================

    /**
     * @brief Computes all quality metrics from a PolyMesh object.
     * @param mesh The analyzed PolyMesh object
     * @return A new MeshQuality instance with computed metrics
     * @throws std::runtime_error if mesh has not been analyzed
     */
    static MeshQuality fromMesh(const PolyMesh& mesh);

    // =========================================================================
    // Output Methods
    // =========================================================================

    /**
     * @brief Prints a formatted summary of the quality metrics.
     */
    void printSummary() const;

private:
    // =========================================================================
    // Internal Computation Methods
    // =========================================================================

    /**
     * @brief Computes the min/max cell volume ratio.
     */
    static double computeVolumeRatio(const PolyMesh& mesh);

    /**
     * @brief Computes skewness and aspect ratio for all cells.
     * @return A pair of vectors: (skewness, aspectRatio)
     */
    static std::pair<std::vector<double>, std::vector<double>>
    computeGeometricMetrics(const PolyMesh& mesh);

    /**
     * @brief Computes skewness and aspect ratio for a triangle.
     * @param nodes Array of 3 node coordinates (2D, x and y only)
     * @return A pair: (skewness, aspectRatio)
     */
    static std::pair<double, double>
    computeTriangleMetrics(const std::vector<std::array<double, 2>>& nodes);

    /**
     * @brief Computes skewness and aspect ratio for a quadrilateral.
     * @param nodes Array of 4 node coordinates (2D, x and y only)
     * @return A pair: (skewness, aspectRatio)
     */
    static std::pair<double, double>
    computeQuadMetrics(const std::vector<std::array<double, 2>>& nodes);

    /**
     * @brief Computes the maximum non-orthogonality for each cell.
     */
    static std::vector<double> computeNonOrthogonality(const PolyMesh& mesh);

    /**
     * @brief Checks for topological issues in the mesh.
     */
    static std::vector<std::string> checkConnectivity(const PolyMesh& mesh);
};

}  // namespace fvm
