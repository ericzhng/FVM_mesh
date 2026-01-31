#include "mesh_generator.hpp"
#include <gmsh.h>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <set>
#include <stdexcept>
#include <unordered_set>

namespace fvm {

MeshGenerator::MeshGenerator(const std::vector<int>& surfaceTags,
                             const std::string& outputDir)
    : surfaceTags_(surfaceTags), outputDir_(outputDir) {
    ensureOutputDir();
}

MeshGenerator::MeshGenerator(int surfaceTag, const std::string& outputDir)
    : surfaceTags_({surfaceTag}), outputDir_(outputDir) {
    ensureOutputDir();
}

void MeshGenerator::ensureOutputDir() {
    std::filesystem::create_directories(outputDir_);
}

void MeshGenerator::generate(const std::map<int, MeshParams>& meshParams,
                             const std::string& filename) {
    // Apply mesh parameters for each surface
    for (int surfaceTag : surfaceTags_) {
        auto it = meshParams.find(surfaceTag);
        if (it != meshParams.end()) {
            applyMeshParameters(surfaceTag, it->second);
        }
    }

    // Set up physical groups
    setupPhysicalGroups();

    // Generate 2D mesh
    gmsh::model::mesh::generate(2);

    // Save mesh to Gmsh format
    saveMesh(filename);

    // Extract mesh data for further processing
    extractMeshData();

    std::cout << "Mesh generation complete." << std::endl;
    std::cout << "  Nodes: " << meshData_.nodes.size() << std::endl;
    std::cout << "  Cells: " << meshData_.cells.size() << std::endl;
}

void MeshGenerator::applyMeshParameters(int surfaceTag, const MeshParams& params) {
    switch (params.meshType) {
        case MeshType::Structured:
            setStructuredMesh(surfaceTag, params.charLength);
            break;
        case MeshType::Quads:
            gmsh::model::mesh::setRecombine(2, surfaceTag);
            break;
        case MeshType::Triangles:
            // Default meshing, no special settings needed
            break;
    }
}

void MeshGenerator::setStructuredMesh(int surfaceTag, double charLength) {
    gmsh::model::mesh::setTransfiniteSurface(surfaceTag);
    gmsh::model::mesh::setRecombine(2, surfaceTag);

    // Get boundary curves
    std::vector<std::pair<int, int>> boundaryCurves;
    gmsh::model::getBoundary({{2, surfaceTag}}, boundaryCurves, false, false, false);

    if (boundaryCurves.size() != 4) {
        throw std::runtime_error(
            "Structured mesh is only supported for geometries with 4 boundary curves.");
    }

    // Get bounding box for calculating number of divisions
    double minX, minY, minZ, maxX, maxY, maxZ;
    gmsh::model::getBoundingBox(2, surfaceTag, minX, minY, minZ, maxX, maxY, maxZ);

    double dx = maxX - minX;
    double dy = maxY - minY;
    int nx = static_cast<int>(dx / charLength);
    int ny = static_cast<int>(dy / charLength);

    // Classify curves and set transfinite
    auto [hCurves, vCurves] = classifyBoundaryCurves(boundaryCurves);

    for (int curveTag : hCurves) {
        gmsh::model::mesh::setTransfiniteCurve(curveTag, nx + 1);
    }
    for (int curveTag : vCurves) {
        gmsh::model::mesh::setTransfiniteCurve(curveTag, ny + 1);
    }
}

std::pair<std::vector<int>, std::vector<int>>
MeshGenerator::classifyBoundaryCurves(
    const std::vector<std::pair<int, int>>& boundaryCurves) const {

    std::vector<int> hCurves, vCurves;

    for (const auto& dimTag : boundaryCurves) {
        int curveTag = dimTag.second;

        // Get curve endpoints
        std::vector<std::pair<int, int>> pointTags;
        gmsh::model::getBoundary({{1, curveTag}}, pointTags, false, false, false);

        if (pointTags.size() < 2) continue;

        int pStartTag = pointTags[0].second;
        int pEndTag = pointTags[1].second;

        std::vector<double> coordStart, coordEnd;
        std::vector<double> paramCoord;  // unused but required
        gmsh::model::getValue(0, pStartTag, paramCoord, coordStart);
        gmsh::model::getValue(0, pEndTag, paramCoord, coordEnd);

        // Classify by orientation
        if (std::abs(coordStart[1] - coordEnd[1]) < 1e-6) {
            hCurves.push_back(curveTag);
        } else {
            vCurves.push_back(curveTag);
        }
    }

    return {hCurves, vCurves};
}

void MeshGenerator::setupPhysicalGroups() {
    // Collect all boundary curves
    std::set<int> allBoundaryCurves;
    for (int surfaceTag : surfaceTags_) {
        std::vector<std::pair<int, int>> boundary;
        gmsh::model::getBoundary({{2, surfaceTag}}, boundary, false, false, false);
        for (const auto& dimTag : boundary) {
            allBoundaryCurves.insert(dimTag.second);
        }
    }

    // Find curves already in physical groups
    std::set<int> curvesInGroups;
    std::vector<std::pair<int, int>> physicalGroups;
    gmsh::model::getPhysicalGroups(physicalGroups, 1);

    for (const auto& [dim, tag] : physicalGroups) {
        std::vector<int> entities;
        gmsh::model::getEntitiesForPhysicalGroup(dim, tag, entities);
        for (int e : entities) {
            curvesInGroups.insert(e);
        }
    }

    // Add untagged curves to "unnamed" group
    std::vector<int> untaggedCurves;
    for (int curve : allBoundaryCurves) {
        if (curvesInGroups.find(curve) == curvesInGroups.end()) {
            untaggedCurves.push_back(curve);
        }
    }
    if (!untaggedCurves.empty()) {
        gmsh::model::addPhysicalGroup(1, untaggedCurves, -1, "unnamed");
    }

    // Handle 2D physical groups (surfaces)
    std::set<int> surfacesInGroups;
    gmsh::model::getPhysicalGroups(physicalGroups, 2);

    for (const auto& [dim, tag] : physicalGroups) {
        std::vector<int> entities;
        gmsh::model::getEntitiesForPhysicalGroup(dim, tag, entities);
        for (int e : entities) {
            surfacesInGroups.insert(e);
        }
    }

    // Add untagged surfaces to "fluid" group
    std::vector<int> untaggedSurfaces;
    for (int surface : surfaceTags_) {
        if (surfacesInGroups.find(surface) == surfacesInGroups.end()) {
            untaggedSurfaces.push_back(surface);
        }
    }
    if (!untaggedSurfaces.empty()) {
        gmsh::model::addPhysicalGroup(2, untaggedSurfaces, -1, "fluid");
    }
}

void MeshGenerator::saveMesh(const std::string& filename) {
    std::string mshFile = outputDir_ + "/" + filename;
    gmsh::write(mshFile);
    std::cout << "Mesh saved to: " << mshFile << std::endl;
}

void MeshGenerator::extractMeshData() {
    extractNodes();
    extractCells();
    extractPhysicalGroups();
    extractFaces();
}

void MeshGenerator::extractNodes() {
    std::vector<std::size_t> nodeTags;
    std::vector<double> nodeCoords;
    std::vector<double> parametricCoords;

    gmsh::model::mesh::getNodes(nodeTags, nodeCoords, parametricCoords);

    meshData_.nodes.clear();
    meshData_.nodeIds.clear();
    meshData_.nodes.reserve(nodeTags.size());
    meshData_.nodeIds.reserve(nodeTags.size());

    for (std::size_t i = 0; i < nodeTags.size(); ++i) {
        meshData_.nodeIds.push_back(nodeTags[i]);
        meshData_.nodes.push_back({
            nodeCoords[3 * i],
            nodeCoords[3 * i + 1],
            nodeCoords[3 * i + 2]
        });
    }
}

void MeshGenerator::extractCells() {
    // Create node tag to index map
    std::unordered_map<std::size_t, std::size_t> nodeMap;
    for (std::size_t i = 0; i < meshData_.nodeIds.size(); ++i) {
        nodeMap[meshData_.nodeIds[i]] = i;
    }

    meshData_.cells.clear();
    meshData_.cellTypes.clear();

    for (int surfaceTag : surfaceTags_) {
        std::vector<int> elemTypes;
        std::vector<std::vector<std::size_t>> elemTags;
        std::vector<std::vector<std::size_t>> elemNodeTags;

        gmsh::model::mesh::getElements(elemTypes, elemTags, elemNodeTags, 2, surfaceTag);

        for (std::size_t i = 0; i < elemTypes.size(); ++i) {
            int elemType = elemTypes[i];

            // Get element properties
            std::string name;
            int dim, order, numNodes, numPrimaryNodes;
            std::vector<double> localNodeCoords;
            gmsh::model::mesh::getElementProperties(
                elemType, name, dim, order, numNodes, localNodeCoords, numPrimaryNodes);

            std::size_t numElements = elemTags[i].size();
            const auto& allNodeTags = elemNodeTags[i];

            for (std::size_t j = 0; j < numElements; ++j) {
                CellConnectivity cell;
                cell.reserve(numNodes);

                for (int k = 0; k < numNodes; ++k) {
                    std::size_t nodeTag = allNodeTags[j * numNodes + k];
                    cell.push_back(nodeMap[nodeTag]);
                }

                meshData_.cells.push_back(std::move(cell));
                meshData_.cellTypes.push_back(getVTKCellType(numNodes));
            }
        }
    }
}

void MeshGenerator::extractPhysicalGroups() {
    meshData_.boundaryGroups.clear();
    meshData_.volumeGroups.clear();

    std::vector<std::pair<int, int>> physicalGroups;
    gmsh::model::getPhysicalGroups(physicalGroups);

    for (const auto& [dim, tag] : physicalGroups) {
        std::string name;
        gmsh::model::getPhysicalName(dim, tag, name);

        std::vector<int> entities;
        gmsh::model::getEntitiesForPhysicalGroup(dim, tag, entities);

        PhysicalGroup group;
        group.dimension = dim;
        group.tag = tag;
        group.name = name.empty() ? "group_" + std::to_string(tag) : name;
        group.entities = entities;

        if (dim == 1) {
            meshData_.boundaryGroups[group.name] = group;
        } else if (dim == 2) {
            meshData_.volumeGroups[group.name] = group;
        }
    }
}

void MeshGenerator::extractFaces() {
    // For 2D meshes, faces are edges
    // Build edge-to-cell connectivity

    // Create node tag to index map
    std::unordered_map<std::size_t, std::size_t> nodeMap;
    for (std::size_t i = 0; i < meshData_.nodeIds.size(); ++i) {
        nodeMap[meshData_.nodeIds[i]] = i;
    }

    // Map from edge (ordered pair) to cells sharing it
    std::map<std::pair<std::size_t, std::size_t>, std::vector<std::size_t>> edgeToCells;

    for (std::size_t cellIdx = 0; cellIdx < meshData_.cells.size(); ++cellIdx) {
        const auto& cell = meshData_.cells[cellIdx];
        std::size_t n = cell.size();

        for (std::size_t i = 0; i < n; ++i) {
            std::size_t n1 = cell[i];
            std::size_t n2 = cell[(i + 1) % n];

            // Create ordered edge key
            auto edge = std::minmax(n1, n2);
            edgeToCells[edge].push_back(cellIdx);
        }
    }

    meshData_.internalFaces.clear();
    meshData_.boundaryFaces.clear();
    meshData_.boundaryFaceLabels.clear();

    // Get boundary elements for labeling
    std::map<std::pair<std::size_t, std::size_t>, std::string> boundaryEdgeLabels;

    for (const auto& [name, group] : meshData_.boundaryGroups) {
        for (int entityTag : group.entities) {
            std::vector<int> elemTypes;
            std::vector<std::vector<std::size_t>> elemTags;
            std::vector<std::vector<std::size_t>> elemNodeTags;

            gmsh::model::mesh::getElements(elemTypes, elemTags, elemNodeTags, 1, entityTag);

            for (std::size_t i = 0; i < elemTypes.size(); ++i) {
                // Line elements have 2 nodes
                const auto& nodeTags = elemNodeTags[i];
                for (std::size_t j = 0; j + 1 < nodeTags.size(); j += 2) {
                    std::size_t n1 = nodeMap[nodeTags[j]];
                    std::size_t n2 = nodeMap[nodeTags[j + 1]];
                    auto edge = std::minmax(n1, n2);
                    boundaryEdgeLabels[edge] = name;
                }
            }
        }
    }

    // Classify edges
    for (const auto& [edge, cells] : edgeToCells) {
        if (cells.size() == 2) {
            // Internal face
            meshData_.internalFaces.push_back({edge.first, edge.second});
        } else if (cells.size() == 1) {
            // Boundary face
            meshData_.boundaryFaces.push_back({edge.first, edge.second});

            auto it = boundaryEdgeLabels.find(edge);
            if (it != boundaryEdgeLabels.end()) {
                meshData_.boundaryFaceLabels.push_back(it->second);
            } else {
                meshData_.boundaryFaceLabels.push_back("unnamed");
            }
        }
    }
}

}  // namespace fvm
