#pragma once

#include "mesh_processing_export.hpp"
#include "mesh_types.hpp"
#include <string>

namespace fvm {

/**
 * @brief Utility class for writing mesh data to VTK format.
 *
 * Supports both legacy VTK (.vtk) and XML VTU (.vtu) formats.
 * These formats are widely used in parallel FVM solvers and
 * can be visualized with ParaView.
 */
class MESH_PROCESSING_API VTKWriter {
public:
    /**
     * @brief Write mesh data to legacy VTK format (.vtk).
     * @param mesh The mesh data to write
     * @param filename Output filename
     * @param binary Whether to use binary format (default: false for ASCII)
     */
    static void writeVTK(const MeshData& mesh,
                         const std::string& filename,
                         bool binary = false);

    /**
     * @brief Write mesh data to XML VTU format (.vtu).
     * @param mesh The mesh data to write
     * @param filename Output filename
     * @param binary Whether to use binary format (default: false for ASCII)
     */
    static void writeVTU(const MeshData& mesh,
                         const std::string& filename,
                         bool binary = false);

    /**
     * @brief Write mesh data to OpenFOAM polyMesh format.
     * @param mesh The mesh data to write
     * @param outputDir Output directory (will create polyMesh subdirectory)
     */
    static void writeOpenFOAM(const MeshData& mesh,
                              const std::string& outputDir);

    /**
     * @brief Write boundary conditions to a separate file.
     * @param mesh The mesh data
     * @param filename Output filename
     */
    static void writeBoundaryInfo(const MeshData& mesh,
                                  const std::string& filename);

private:
    /// Write VTK header
    static void writeVTKHeader(std::ostream& os, const std::string& title);

    /// Write points section
    static void writeVTKPoints(std::ostream& os, const MeshData& mesh);

    /// Write cells section
    static void writeVTKCells(std::ostream& os, const MeshData& mesh);

    /// Write cell types
    static void writeVTKCellTypes(std::ostream& os, const MeshData& mesh);

    /// Write cell data (boundary labels, etc.)
    static void writeVTKCellData(std::ostream& os, const MeshData& mesh);
};

}  // namespace fvm
