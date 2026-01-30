#pragma once

/**
 * @file poly_mesh.hpp
 * @brief Comprehensive data structure for 2D/3D unstructured polygonal meshes.
 *
 * The PolyMesh class is designed to be a stand-alone, data-centric container
 * for mesh information required for finite volume method (FVM) solvers.
 */

#include <array>
#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace fvm {

// Forward declarations
class MeshQuality;

/**
 * @brief Properties of an element type (e.g., triangle, quad, tetrahedron).
 */
struct ElementTypeProperties {
    std::string name;
    int numNodes;
};

/**
 * @brief Information about a physical group (boundary or volume region).
 */
struct PhysicalGroup {
    int dimension;
    int tag;
    std::string name;
    std::vector<int> entities;
};

/**
 * @brief A comprehensive data structure for unstructured polygonal meshes.
 *
 * This class holds all fundamental mesh data including topology and computed
 * geometric properties. It can be initialized from a Gmsh .msh file and
 * analyzed to prepare for numerical simulations.
 */
class PolyMesh {
public:
    // =========================================================================
    // Core Mesh Properties
    // =========================================================================

    /// Spatial dimension of the mesh (2 or 3)
    int dimension = 0;

    /// Total number of nodes (vertices)
    std::size_t nNodes = 0;

    /// Total number of cells (elements)
    std::size_t nCells = 0;

    // =========================================================================
    // Topology Data (read or defined)
    // =========================================================================

    /// Node coordinates: shape (nNodes, 3)
    std::vector<std::array<double, 3>> nodeCoords;

    /// Cell-node connectivity: jagged array where each inner vector contains
    /// the node indices for a single cell
    std::vector<std::vector<std::size_t>> cellNodeConnectivity;

    /// Element type ID for each cell (Gmsh type codes)
    std::vector<int> cellElementTypes;

    /// Map from element type ID to properties
    std::unordered_map<int, ElementTypeProperties> elementTypeProperties;

    // =========================================================================
    // Topology Data (derived)
    // =========================================================================

    /// Faces for each cell: cellFaceNodes[cell][face] = vector of node indices
    std::vector<std::vector<std::vector<std::size_t>>> cellFaceNodes;

    /// Neighbor cell indices for each face of each cell (-1 for boundary)
    /// Shape: (nCells, maxFacesPerCell)
    std::vector<std::vector<int>> cellNeighbors;

    /// Physical tag for each face of each cell (0 for interior faces)
    /// Shape: (nCells, maxFacesPerCell)
    std::vector<std::vector<int>> cellFaceTags;

    // =========================================================================
    // Boundary Data
    // =========================================================================

    /// Node indices for each unique boundary face
    std::vector<std::vector<std::size_t>> boundaryFaceNodes;

    /// Physical tag for each unique boundary face
    std::vector<int> boundaryFaceTags;

    /// Mapping from physical group names to their integer tags
    std::unordered_map<std::string, int> boundaryPatchMap;

    // =========================================================================
    // Computed Geometric Properties
    // =========================================================================

    /// Geometric center of each cell: shape (nCells, 3)
    std::vector<std::array<double, 3>> cellCentroids;

    /// Volume (3D) or area (2D) of each cell
    std::vector<double> cellVolumes;

    /// Geometric center of each face: shape (nCells, maxFaces, 3)
    std::vector<std::vector<std::array<double, 3>>> cellFaceMidpoints;

    /// Outward normal vector of each face: shape (nCells, maxFaces, 3)
    std::vector<std::vector<std::array<double, 3>>> cellFaceNormals;

    /// Area (3D) or length (2D) of each face: shape (nCells, maxFaces)
    std::vector<std::vector<double>> cellFaceAreas;

    /// Distance from each face's midpoint to its cell's centroid
    std::vector<std::vector<double>> faceToCentroidDistances;

    // =========================================================================
    // Quality Metrics
    // =========================================================================

    /// Mesh quality metrics (computed on demand)
    std::unique_ptr<MeshQuality> quality;

    // =========================================================================
    // Constructors and Factory Methods
    // =========================================================================

    PolyMesh();
    virtual ~PolyMesh();

    // Allow move but prevent copy (mesh data can be large)
    PolyMesh(PolyMesh&&) noexcept;
    PolyMesh& operator=(PolyMesh&&) noexcept;
    PolyMesh(const PolyMesh&) = delete;
    PolyMesh& operator=(const PolyMesh&) = delete;

    /**
     * @brief Creates a PolyMesh from a Gmsh .msh file.
     * @param mshFile Path to the .msh file
     * @param gmshVerbose Verbosity level for Gmsh API (0-10)
     * @return A new PolyMesh instance populated with data from the file
     */
    static PolyMesh fromGmsh(const std::string& mshFile, int gmshVerbose = 0);

    /**
     * @brief Creates a structured quadrilateral mesh for testing.
     * @param nx Number of cells in x-direction
     * @param ny Number of cells in y-direction
     * @return A new, analyzed PolyMesh instance
     */
    static PolyMesh createStructuredQuadMesh(int nx, int ny);

    // =========================================================================
    // Public API
    // =========================================================================

    /**
     * @brief Computes all derived topological and geometric properties.
     *
     * This method orchestrates computations to build the full mesh data
     * structure. Should be called after reading the mesh file.
     * Can be called again after reordering to recompute derived properties.
     */
    void analyzeMesh();

    /**
     * @brief Clears all derived data, forcing recomputation on next analyze.
     *
     * This is useful after reordering cells or nodes, as the derived
     * properties (neighbors, centroids, etc.) become invalid.
     */
    void clearDerivedData();

    /**
     * @brief Prints a formatted summary report of the mesh analysis.
     */
    void printSummary() const;

    /**
     * @brief Reads mesh data from a Gmsh .msh file.
     * @param mshFile Path to the .msh file
     * @param gmshVerbose Verbosity level for Gmsh API
     */
    void readGmsh(const std::string& mshFile, int gmshVerbose = 0);

    /**
     * @brief Checks if the mesh has been analyzed.
     */
    bool isAnalyzed() const { return isAnalyzed_; }

protected:
    // =========================================================================
    // Internal State
    // =========================================================================

    bool isAnalyzed_ = false;
    std::unordered_map<std::size_t, std::size_t> tagToIndex_;

    // =========================================================================
    // Mesh I/O Methods
    // =========================================================================

    void readNodes();
    void readElements();
    void readPhysicalGroups();

    // =========================================================================
    // Topology and Connectivity Computations
    // =========================================================================

    void extractCellFaces();
    std::vector<std::vector<std::size_t>> getFacesForCell(
        const std::vector<std::size_t>& conn) const;
    void computeFaceTopology();

    // =========================================================================
    // Geometric Property Computations
    // =========================================================================

    void computeCellCentroids();
    void computeFaceProperties();
    void compute2DFaceMetrics(std::size_t ci, std::size_t fi,
                              const std::vector<std::array<double, 3>>& nodes);
    void compute3DFaceMetrics(std::size_t ci, std::size_t fi,
                              const std::vector<std::array<double, 3>>& nodes);
    void orientFaceNormals();
    void computeFaceToCentroidDist();
    void computeCellVolumes();
    void compute2DCellVolumes();
    void compute3DCellVolumes();

    // =========================================================================
    // Helper Methods
    // =========================================================================

    int getMeshDimension(const std::vector<int>& elemTypes) const;
    void printGeneralInfo() const;
    void printGeometricProperties() const;
    void printCellGeometry() const;
    void printCellTypeDistribution() const;
    void printStatLine(const std::string& name, const std::vector<double>& data) const;
};

}  // namespace fvm
