#include "vtk_writer.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>

namespace fvm {

void VTKWriter::writeVTK(const MeshData& mesh,
                         const std::string& filename,
                         bool binary) {
    std::ofstream ofs(filename);
    if (!ofs) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }

    ofs << std::fixed << std::setprecision(10);

    // Header
    writeVTKHeader(ofs, "FVM Mesh");

    if (binary) {
        ofs << "BINARY\n";
    } else {
        ofs << "ASCII\n";
    }

    ofs << "DATASET UNSTRUCTURED_GRID\n";

    // Points
    writeVTKPoints(ofs, mesh);

    // Cells
    writeVTKCells(ofs, mesh);

    // Cell types
    writeVTKCellTypes(ofs, mesh);

    // Cell data (boundary info, etc.)
    writeVTKCellData(ofs, mesh);

    ofs.close();
    std::cout << "VTK file written: " << filename << std::endl;
}

void VTKWriter::writeVTU(const MeshData& mesh,
                         const std::string& filename,
                         bool binary) {
    std::ofstream ofs(filename);
    if (!ofs) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }

    ofs << std::fixed << std::setprecision(10);

    // XML header
    ofs << "<?xml version=\"1.0\"?>\n";
    ofs << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\">\n";
    ofs << "  <UnstructuredGrid>\n";
    ofs << "    <Piece NumberOfPoints=\"" << mesh.nodes.size()
        << "\" NumberOfCells=\"" << mesh.cells.size() << "\">\n";

    // Points
    ofs << "      <Points>\n";
    ofs << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (const auto& node : mesh.nodes) {
        ofs << "          " << node[0] << " " << node[1] << " " << node[2] << "\n";
    }
    ofs << "        </DataArray>\n";
    ofs << "      </Points>\n";

    // Cells
    ofs << "      <Cells>\n";

    // Connectivity
    ofs << "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n";
    for (const auto& cell : mesh.cells) {
        ofs << "          ";
        for (std::size_t nodeIdx : cell) {
            ofs << nodeIdx << " ";
        }
        ofs << "\n";
    }
    ofs << "        </DataArray>\n";

    // Offsets
    ofs << "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n";
    std::size_t offset = 0;
    ofs << "          ";
    for (const auto& cell : mesh.cells) {
        offset += cell.size();
        ofs << offset << " ";
    }
    ofs << "\n";
    ofs << "        </DataArray>\n";

    // Types
    ofs << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    ofs << "          ";
    for (int cellType : mesh.cellTypes) {
        ofs << cellType << " ";
    }
    ofs << "\n";
    ofs << "        </DataArray>\n";

    ofs << "      </Cells>\n";

    // Cell Data
    ofs << "      <CellData>\n";

    // Cell IDs
    ofs << "        <DataArray type=\"Int32\" Name=\"CellID\" format=\"ascii\">\n";
    ofs << "          ";
    for (std::size_t i = 0; i < mesh.cells.size(); ++i) {
        ofs << i << " ";
    }
    ofs << "\n";
    ofs << "        </DataArray>\n";

    ofs << "      </CellData>\n";

    // Point Data
    ofs << "      <PointData>\n";

    // Point IDs
    ofs << "        <DataArray type=\"Int32\" Name=\"PointID\" format=\"ascii\">\n";
    ofs << "          ";
    for (std::size_t i = 0; i < mesh.nodes.size(); ++i) {
        ofs << i << " ";
    }
    ofs << "\n";
    ofs << "        </DataArray>\n";

    ofs << "      </PointData>\n";

    ofs << "    </Piece>\n";
    ofs << "  </UnstructuredGrid>\n";
    ofs << "</VTKFile>\n";

    ofs.close();
    std::cout << "VTU file written: " << filename << std::endl;
}

void VTKWriter::writeOpenFOAM(const MeshData& mesh,
                              const std::string& outputDir) {
    std::string polyMeshDir = outputDir + "/constant/polyMesh";
    std::filesystem::create_directories(polyMeshDir);

    auto writeFoamHeader = [](std::ofstream& ofs, const std::string& objectClass,
                              const std::string& objectName) {
        ofs << "FoamFile\n";
        ofs << "{\n";
        ofs << "    version     2.0;\n";
        ofs << "    format      ascii;\n";
        ofs << "    class       " << objectClass << ";\n";
        ofs << "    object      " << objectName << ";\n";
        ofs << "}\n\n";
    };

    // Write points
    {
        std::ofstream ofs(polyMeshDir + "/points");
        writeFoamHeader(ofs, "vectorField", "points");

        ofs << std::fixed << std::setprecision(10);
        ofs << mesh.nodes.size() << "\n(\n";
        for (const auto& node : mesh.nodes) {
            ofs << "(" << node[0] << " " << node[1] << " " << node[2] << ")\n";
        }
        ofs << ")\n";
    }

    // For 2D meshes in OpenFOAM, we need to extrude to 3D
    // This is a simplified 2D export (single layer)
    // Note: A proper 2D OpenFOAM mesh would need extrusion

    // Write faces (for 2D, each cell edge becomes a face)
    std::vector<std::vector<std::size_t>> allFaces;
    std::vector<int> owner;
    std::vector<int> neighbour;

    // Build face data from cells
    std::map<std::pair<std::size_t, std::size_t>, std::size_t> edgeToFaceIdx;

    for (std::size_t cellIdx = 0; cellIdx < mesh.cells.size(); ++cellIdx) {
        const auto& cell = mesh.cells[cellIdx];
        std::size_t n = cell.size();

        for (std::size_t i = 0; i < n; ++i) {
            std::size_t n1 = cell[i];
            std::size_t n2 = cell[(i + 1) % n];
            auto edge = std::minmax(n1, n2);

            auto it = edgeToFaceIdx.find(edge);
            if (it == edgeToFaceIdx.end()) {
                // New face
                std::size_t faceIdx = allFaces.size();
                allFaces.push_back({n1, n2});
                owner.push_back(static_cast<int>(cellIdx));
                edgeToFaceIdx[edge] = faceIdx;
            } else {
                // Existing face - this cell is the neighbour
                neighbour.resize(allFaces.size(), -1);
                neighbour[it->second] = static_cast<int>(cellIdx);
            }
        }
    }

    // Separate internal and boundary faces
    std::vector<std::size_t> internalFaceIndices;
    std::vector<std::size_t> boundaryFaceIndices;

    neighbour.resize(allFaces.size(), -1);
    for (std::size_t i = 0; i < allFaces.size(); ++i) {
        if (neighbour[i] >= 0) {
            internalFaceIndices.push_back(i);
        } else {
            boundaryFaceIndices.push_back(i);
        }
    }

    // Write faces file
    {
        std::ofstream ofs(polyMeshDir + "/faces");
        writeFoamHeader(ofs, "faceList", "faces");

        std::size_t totalFaces = internalFaceIndices.size() + boundaryFaceIndices.size();
        ofs << totalFaces << "\n(\n";

        // Internal faces first
        for (std::size_t idx : internalFaceIndices) {
            const auto& face = allFaces[idx];
            ofs << face.size() << "(";
            for (std::size_t j = 0; j < face.size(); ++j) {
                if (j > 0) ofs << " ";
                ofs << face[j];
            }
            ofs << ")\n";
        }

        // Boundary faces
        for (std::size_t idx : boundaryFaceIndices) {
            const auto& face = allFaces[idx];
            ofs << face.size() << "(";
            for (std::size_t j = 0; j < face.size(); ++j) {
                if (j > 0) ofs << " ";
                ofs << face[j];
            }
            ofs << ")\n";
        }

        ofs << ")\n";
    }

    // Write owner
    {
        std::ofstream ofs(polyMeshDir + "/owner");
        writeFoamHeader(ofs, "labelList", "owner");

        std::size_t totalFaces = internalFaceIndices.size() + boundaryFaceIndices.size();
        ofs << totalFaces << "\n(\n";

        for (std::size_t idx : internalFaceIndices) {
            ofs << owner[idx] << "\n";
        }
        for (std::size_t idx : boundaryFaceIndices) {
            ofs << owner[idx] << "\n";
        }

        ofs << ")\n";
    }

    // Write neighbour (only for internal faces)
    {
        std::ofstream ofs(polyMeshDir + "/neighbour");
        writeFoamHeader(ofs, "labelList", "neighbour");

        ofs << internalFaceIndices.size() << "\n(\n";
        for (std::size_t idx : internalFaceIndices) {
            ofs << neighbour[idx] << "\n";
        }
        ofs << ")\n";
    }

    // Write boundary
    {
        std::ofstream ofs(polyMeshDir + "/boundary");
        writeFoamHeader(ofs, "polyBoundaryMesh", "boundary");

        // Group boundary faces by label
        std::map<std::string, std::vector<std::size_t>> boundaryPatches;
        for (std::size_t i = 0; i < boundaryFaceIndices.size(); ++i) {
            std::string label = (i < mesh.boundaryFaceLabels.size())
                                ? mesh.boundaryFaceLabels[i]
                                : "unnamed";
            boundaryPatches[label].push_back(i);
        }

        ofs << boundaryPatches.size() << "\n(\n";

        std::size_t startFace = internalFaceIndices.size();
        for (const auto& [name, faces] : boundaryPatches) {
            ofs << "    " << name << "\n";
            ofs << "    {\n";
            ofs << "        type            patch;\n";
            ofs << "        nFaces          " << faces.size() << ";\n";
            ofs << "        startFace       " << startFace << ";\n";
            ofs << "    }\n";
            startFace += faces.size();
        }

        ofs << ")\n";
    }

    std::cout << "OpenFOAM polyMesh written to: " << polyMeshDir << std::endl;
}

void VTKWriter::writeBoundaryInfo(const MeshData& mesh,
                                  const std::string& filename) {
    std::ofstream ofs(filename);
    if (!ofs) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }

    ofs << "# Boundary Information\n";
    ofs << "# Format: boundary_name num_faces entity_tags...\n\n";

    for (const auto& [name, group] : mesh.boundaryGroups) {
        ofs << name << " " << group.entities.size();
        for (int tag : group.entities) {
            ofs << " " << tag;
        }
        ofs << "\n";
    }

    ofs << "\n# Boundary Face List\n";
    ofs << "# Format: face_index node1 node2 boundary_name\n\n";

    for (std::size_t i = 0; i < mesh.boundaryFaces.size(); ++i) {
        const auto& face = mesh.boundaryFaces[i];
        std::string label = (i < mesh.boundaryFaceLabels.size())
                           ? mesh.boundaryFaceLabels[i]
                           : "unnamed";
        ofs << i << " " << face[0] << " " << face[1] << " " << label << "\n";
    }

    ofs.close();
    std::cout << "Boundary info written to: " << filename << std::endl;
}

void VTKWriter::writeVTKHeader(std::ostream& os, const std::string& title) {
    os << "# vtk DataFile Version 3.0\n";
    os << title << "\n";
}

void VTKWriter::writeVTKPoints(std::ostream& os, const MeshData& mesh) {
    os << "POINTS " << mesh.nodes.size() << " double\n";
    for (const auto& node : mesh.nodes) {
        os << node[0] << " " << node[1] << " " << node[2] << "\n";
    }
}

void VTKWriter::writeVTKCells(std::ostream& os, const MeshData& mesh) {
    // Calculate total size (num_nodes_per_cell + node_indices for each cell)
    std::size_t totalSize = 0;
    for (const auto& cell : mesh.cells) {
        totalSize += 1 + cell.size();  // 1 for count, rest for indices
    }

    os << "CELLS " << mesh.cells.size() << " " << totalSize << "\n";
    for (const auto& cell : mesh.cells) {
        os << cell.size();
        for (std::size_t nodeIdx : cell) {
            os << " " << nodeIdx;
        }
        os << "\n";
    }
}

void VTKWriter::writeVTKCellTypes(std::ostream& os, const MeshData& mesh) {
    os << "CELL_TYPES " << mesh.cells.size() << "\n";
    for (int cellType : mesh.cellTypes) {
        os << cellType << "\n";
    }
}

void VTKWriter::writeVTKCellData(std::ostream& os, const MeshData& mesh) {
    os << "CELL_DATA " << mesh.cells.size() << "\n";

    // Cell IDs
    os << "SCALARS CellID int 1\n";
    os << "LOOKUP_TABLE default\n";
    for (std::size_t i = 0; i < mesh.cells.size(); ++i) {
        os << i << "\n";
    }
}

}  // namespace fvm
