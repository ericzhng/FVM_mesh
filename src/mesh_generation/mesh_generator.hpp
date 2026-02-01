#pragma once

#include "mesh_generation_export.hpp"
#include "mesh_types.hpp"
#include <map>
#include <string>
#include <vector>

namespace fvm {

/**
 * @brief A class to generate 2D meshes using Gmsh.
 *
 * This class handles the generation of structured, triangular, and quadrilateral
 * meshes, and provides methods for saving the generated mesh to various formats.
 */
class MESH_GENERATION_API MeshGenerator {
public:
    /**
     * @brief Construct a new MeshGenerator object.
     * @param surfaceTags List of surface tags to be meshed
     * @param outputDir Directory to save output files
     */
    MeshGenerator(const std::vector<int>& surfaceTags,
                  const std::string& outputDir = ".");

    /**
     * @brief Construct with a single surface tag.
     * @param surfaceTag Surface tag to be meshed
     * @param outputDir Directory to save output files
     */
    MeshGenerator(int surfaceTag, const std::string& outputDir = ".");

    /**
     * @brief Generate the mesh for the specified surfaces.
     * @param meshParams Map of surface tags to mesh parameters
     * @param filename Output filename (without path)
     */
    void generate(const std::map<int, MeshParams>& meshParams,
                  const std::string& filename = "mesh.msh");

    /**
     * @brief Get the extracted mesh data.
     * @return Reference to the mesh data structure
     */
    const MeshData& getMeshData() const { return meshData_; }

    /**
     * @brief Extract mesh data from current Gmsh model.
     * Populates the internal MeshData structure.
     */
    void extractMeshData();

private:
    std::vector<int> surfaceTags_;
    std::string outputDir_;
    MeshData meshData_;

    /// Apply mesh parameters to a surface
    void applyMeshParameters(int surfaceTag, const MeshParams& params);

    /// Set up structured mesh on a surface
    void setStructuredMesh(int surfaceTag, double charLength);

    /// Classify boundary curves into horizontal and vertical
    std::pair<std::vector<int>, std::vector<int>>
    classifyBoundaryCurves(const std::vector<std::pair<int, int>>& boundaryCurves) const;

    /// Set up physical groups for boundaries and surfaces
    void setupPhysicalGroups();

    /// Save mesh to Gmsh MSH format
    void saveMesh(const std::string& filename);

    /// Extract nodes from Gmsh model
    void extractNodes();

    /// Extract cells from Gmsh model
    void extractCells();

    /// Extract physical groups from Gmsh model
    void extractPhysicalGroups();

    /// Extract face connectivity for FVM
    void extractFaces();

    /// Create output directory if it doesn't exist
    void ensureOutputDir();
};

}  // namespace fvm
