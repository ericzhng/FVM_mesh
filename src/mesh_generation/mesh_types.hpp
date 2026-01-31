#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace fvm {

/// 3D point/vector type
using Point3D = std::array<double, 3>;

/// 2D point/vector type
using Point2D = std::array<double, 2>;

/// Bounding box: {min_x, min_y, min_z, max_x, max_y, max_z}
using BoundingBox = std::array<double, 6>;

/// Cell connectivity (variable number of nodes)
using CellConnectivity = std::vector<std::size_t>;

/// Mesh type enumeration
enum class MeshType {
    Triangles,   // Unstructured triangular mesh
    Quads,       // Unstructured quadrilateral mesh
    Structured   // Structured quadrilateral mesh
};

/// Convert MeshType to string
inline std::string meshTypeToString(MeshType type) {
    switch (type) {
        case MeshType::Triangles:  return "tri";
        case MeshType::Quads:      return "quads";
        case MeshType::Structured: return "structured";
        default:                   return "unknown";
    }
}

/// Parse string to MeshType
inline MeshType stringToMeshType(const std::string& str) {
    if (str == "tri" || str == "triangles") return MeshType::Triangles;
    if (str == "quads" || str == "quad")    return MeshType::Quads;
    if (str == "structured")                return MeshType::Structured;
    return MeshType::Triangles;  // default
}

/// Mesh parameters for a surface
struct MeshParams {
    MeshType meshType = MeshType::Triangles;
    double charLength = 0.1;
};

/// Physical group information
struct PhysicalGroup {
    int dimension;              // 1 for boundary, 2 for surface
    int tag;                    // Gmsh tag
    std::string name;           // Name (e.g., "inlet", "outlet", "wall")
    std::vector<int> entities;  // Entity tags in this group
};

/// Complete mesh data structure for export
struct MeshData {
    std::vector<Point3D> nodes;                              // Node coordinates
    std::vector<std::size_t> nodeIds;                        // Original node IDs
    std::vector<CellConnectivity> cells;                     // Cell connectivity
    std::vector<int> cellTypes;                              // VTK cell types
    std::unordered_map<std::string, PhysicalGroup> boundaryGroups;  // BC groups
    std::unordered_map<std::string, PhysicalGroup> volumeGroups;    // Volume groups

    // Face data for FVM
    std::vector<std::array<std::size_t, 2>> internalFaces;  // Internal face connectivity
    std::vector<std::array<std::size_t, 2>> boundaryFaces;  // Boundary face connectivity
    std::vector<std::string> boundaryFaceLabels;            // BC label for each boundary face
};

/// VTK cell type codes
namespace VTKCellType {
    constexpr int VERTEX = 1;
    constexpr int LINE = 3;
    constexpr int TRIANGLE = 5;
    constexpr int QUAD = 9;
    constexpr int POLYGON = 7;
    constexpr int TETRA = 10;
    constexpr int HEXAHEDRON = 12;
    constexpr int WEDGE = 13;
    constexpr int PYRAMID = 14;
}

/// Map number of nodes to VTK cell type (for 2D cells)
inline int getVTKCellType(std::size_t numNodes) {
    switch (numNodes) {
        case 3: return VTKCellType::TRIANGLE;
        case 4: return VTKCellType::QUAD;
        default: return VTKCellType::POLYGON;
    }
}

}  // namespace fvm
