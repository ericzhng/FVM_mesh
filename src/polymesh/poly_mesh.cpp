#include "poly_mesh.hpp"
#include "mesh_quality.hpp"

#include <gmsh.h>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>

namespace fvm {

// =========================================================================
// Special member functions (must be in .cpp for unique_ptr with forward-declared type)
// =========================================================================

PolyMesh::PolyMesh() = default;
PolyMesh::~PolyMesh() = default;
PolyMesh::PolyMesh(PolyMesh&&) noexcept = default;
PolyMesh& PolyMesh::operator=(PolyMesh&&) noexcept = default;

// =========================================================================
// Factory Methods
// =========================================================================

PolyMesh PolyMesh::fromGmsh(const std::string& mshFile, int gmshVerbose) {
    PolyMesh mesh;
    mesh.readGmsh(mshFile, gmshVerbose);
    return mesh;
}

PolyMesh PolyMesh::createStructuredQuadMesh(int nx, int ny) {
    PolyMesh mesh;
    mesh.dimension = 2;

    int numNodesX = nx + 1;
    int numNodesY = ny + 1;
    mesh.nNodes = static_cast<std::size_t>(numNodesX * numNodesY);
    mesh.nCells = static_cast<std::size_t>(nx * ny);

    // Generate node coordinates
    mesh.nodeCoords.reserve(mesh.nNodes);
    for (int j = 0; j < numNodesY; ++j) {
        for (int i = 0; i < numNodesX; ++i) {
            mesh.nodeCoords.push_back({static_cast<double>(i),
                                       static_cast<double>(j),
                                       0.0});
        }
    }

    // Generate cell connectivity
    mesh.cellNodeConnectivity.reserve(mesh.nCells);
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            std::size_t n0 = static_cast<std::size_t>(j * numNodesX + i);
            std::size_t n1 = static_cast<std::size_t>(j * numNodesX + (i + 1));
            std::size_t n2 = static_cast<std::size_t>((j + 1) * numNodesX + (i + 1));
            std::size_t n3 = static_cast<std::size_t>((j + 1) * numNodesX + i);
            mesh.cellNodeConnectivity.push_back({n0, n1, n2, n3});
        }
    }

    // Add element type information for quads
    mesh.cellElementTypes.resize(mesh.nCells, 0);
    mesh.elementTypeProperties[0] = {"Quad 4", 4};

    // Find and tag boundary faces
    std::map<std::pair<std::size_t, std::size_t>, std::vector<std::size_t>> faceToCells;

    for (std::size_t cellIdx = 0; cellIdx < mesh.nCells; ++cellIdx) {
        const auto& conn = mesh.cellNodeConnectivity[cellIdx];
        for (std::size_t i = 0; i < 4; ++i) {
            auto edge = std::minmax(conn[i], conn[(i + 1) % 4]);
            faceToCells[edge].push_back(cellIdx);
        }
    }

    int bottomTag = 1, rightTag = 2, topTag = 3, leftTag = 4;

    for (const auto& [faceNodes, cells] : faceToCells) {
        if (cells.size() == 1) {
            const auto& node1Coords = mesh.nodeCoords[faceNodes.first];
            const auto& node2Coords = mesh.nodeCoords[faceNodes.second];

            int tag = 0;
            if (std::abs(node1Coords[1]) < 1e-9 && std::abs(node2Coords[1]) < 1e-9) {
                tag = bottomTag;
            } else if (std::abs(node1Coords[0] - nx) < 1e-9 &&
                       std::abs(node2Coords[0] - nx) < 1e-9) {
                tag = rightTag;
            } else if (std::abs(node1Coords[1] - ny) < 1e-9 &&
                       std::abs(node2Coords[1] - ny) < 1e-9) {
                tag = topTag;
            } else if (std::abs(node1Coords[0]) < 1e-9 &&
                       std::abs(node2Coords[0]) < 1e-9) {
                tag = leftTag;
            }

            if (tag > 0) {
                mesh.boundaryFaceNodes.push_back({faceNodes.first, faceNodes.second});
                mesh.boundaryFaceTags.push_back(tag);
            }
        }
    }

    mesh.boundaryPatchMap = {
        {"bottom", bottomTag},
        {"right", rightTag},
        {"top", topTag},
        {"left", leftTag}
    };

    mesh.analyzeMesh();
    return mesh;
}

// =========================================================================
// Public API
// =========================================================================

void PolyMesh::analyzeMesh() {
    if (nCells == 0 || nNodes == 0) {
        throw std::runtime_error(
            "Mesh has no cells or nodes. Load a mesh before analyzing.");
    }

    // Allow re-analysis after data has been modified (e.g., after reordering)
    // Clear the flag to force recomputation
    isAnalyzed_ = false;

    // 1. Compute basic topology and connectivity
    extractCellFaces();
    computeFaceTopology();

    // 2. Compute geometric properties
    computeCellCentroids();
    computeFaceProperties();
    computeFaceToCentroidDist();
    computeCellVolumes();

    isAnalyzed_ = true;
}

void PolyMesh::clearDerivedData() {
    cellFaceNodes.clear();
    cellNeighbors.clear();
    cellFaceTags.clear();
    cellCentroids.clear();
    cellVolumes.clear();
    cellFaceMidpoints.clear();
    cellFaceNormals.clear();
    cellFaceAreas.clear();
    faceToCentroidDistances.clear();
    quality.reset();
    isAnalyzed_ = false;
}

void PolyMesh::printSummary() const {
    if (!isAnalyzed_) {
        std::cout << "Mesh not analyzed. Run analyzeMesh() first.\n";
        return;
    }

    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << std::setw(40 + 10) << "Mesh Analysis Report" << "\n";
    std::cout << std::string(80, '=') << "\n";

    printGeneralInfo();
    printGeometricProperties();
    printCellGeometry();

    // Quality metrics would be printed here if computed
    if (quality) {
        quality->printSummary();
    }

    std::cout << "\n" << std::string(80, '=') << "\n";
}

// =========================================================================
// Mesh I/O Methods
// =========================================================================

void PolyMesh::readGmsh(const std::string& mshFile, int gmshVerbose) {
    gmsh::initialize();
    gmsh::option::setNumber("General.Verbosity", gmshVerbose);

    try {
        gmsh::open(mshFile);
        readNodes();
        readElements();
        readPhysicalGroups();
    } catch (...) {
        gmsh::finalize();
        throw;
    }

    gmsh::finalize();
}

void PolyMesh::readNodes() {
    std::vector<std::size_t> nodeTags;
    std::vector<double> coords;
    std::vector<double> parametricCoords;

    gmsh::model::mesh::getNodes(nodeTags, coords, parametricCoords);

    nNodes = nodeTags.size();
    nodeCoords.resize(nNodes);
    tagToIndex_.clear();

    for (std::size_t i = 0; i < nNodes; ++i) {
        tagToIndex_[nodeTags[i]] = i;
        nodeCoords[i] = {coords[3 * i], coords[3 * i + 1], coords[3 * i + 2]};
    }
}

void PolyMesh::readElements() {
    std::vector<int> elemTypes;
    std::vector<std::vector<std::size_t>> elemTags;
    std::vector<std::vector<std::size_t>> elemNodeTags;

    gmsh::model::mesh::getElements(elemTypes, elemTags, elemNodeTags);

    dimension = getMeshDimension(elemTypes);

    cellNodeConnectivity.clear();
    cellElementTypes.clear();

    for (std::size_t i = 0; i < elemTypes.size(); ++i) {
        int et = elemTypes[i];

        std::string name;
        int dim, order, numNodes, numPrimaryNodes;
        std::vector<double> localNodeCoords;
        gmsh::model::mesh::getElementProperties(et, name, dim, order, numNodes,
                                                 localNodeCoords, numPrimaryNodes);

        if (dim != dimension) {
            continue;  // Skip elements not of the mesh's primary dimension
        }

        elementTypeProperties[et] = {name, numNodes};

        std::size_t numElements = elemTags[i].size();
        const auto& allNodeTags = elemNodeTags[i];

        for (std::size_t j = 0; j < numElements; ++j) {
            std::vector<std::size_t> conn;
            conn.reserve(numNodes);

            for (int k = 0; k < numNodes; ++k) {
                std::size_t nodeTag = allNodeTags[j * numNodes + k];
                conn.push_back(tagToIndex_[nodeTag]);
            }

            cellNodeConnectivity.push_back(std::move(conn));
            cellElementTypes.push_back(et);
        }
    }

    nCells = cellNodeConnectivity.size();
}

void PolyMesh::readPhysicalGroups() {
    int bdim = dimension - 1;
    if (bdim < 0) return;

    std::map<std::set<std::size_t>, std::pair<std::vector<std::size_t>, int>> facesMap;

    std::vector<std::pair<int, int>> physicalGroups;
    gmsh::model::getPhysicalGroups(physicalGroups, bdim);

    for (const auto& [dim, tag] : physicalGroups) {
        std::string name;
        gmsh::model::getPhysicalName(dim, tag, name);
        if (name.empty()) {
            name = std::to_string(tag);
        }
        boundaryPatchMap[name] = tag;

        std::vector<int> entities;
        gmsh::model::getEntitiesForPhysicalGroup(dim, tag, entities);

        for (int ent : entities) {
            std::vector<int> eTypes;
            std::vector<std::vector<std::size_t>> eTags;
            std::vector<std::vector<std::size_t>> eNodeTags;

            try {
                gmsh::model::mesh::getElements(eTypes, eTags, eNodeTags, dim, ent);
            } catch (...) {
                continue;
            }

            for (std::size_t i = 0; i < eTypes.size(); ++i) {
                std::string eName;
                int eDim, eOrder, eNumNodes, ePrimaryNodes;
                std::vector<double> eLocalCoords;
                gmsh::model::mesh::getElementProperties(
                    eTypes[i], eName, eDim, eOrder, eNumNodes, eLocalCoords, ePrimaryNodes);

                const auto& nodeTags = eNodeTags[i];
                for (std::size_t j = 0; j + eNumNodes - 1 < nodeTags.size(); j += eNumNodes) {
                    std::vector<std::size_t> faceNodes;
                    std::set<std::size_t> faceKey;

                    for (int k = 0; k < eNumNodes; ++k) {
                        std::size_t nodeIdx = tagToIndex_[nodeTags[j + k]];
                        faceNodes.push_back(nodeIdx);
                        faceKey.insert(nodeIdx);
                    }

                    auto it = facesMap.find(faceKey);
                    if (it != facesMap.end() && it->second.second != tag) {
                        throw std::runtime_error(
                            "Boundary face assigned to multiple physical groups.");
                    }
                    facesMap[faceKey] = {faceNodes, tag};
                }
            }
        }
    }

    boundaryFaceNodes.clear();
    boundaryFaceTags.clear();

    for (const auto& [key, data] : facesMap) {
        boundaryFaceNodes.push_back(data.first);
        boundaryFaceTags.push_back(data.second);
    }
}

// =========================================================================
// Topology and Connectivity Computations
// =========================================================================

void PolyMesh::extractCellFaces() {
    cellFaceNodes.clear();
    cellFaceNodes.reserve(nCells);

    for (const auto& conn : cellNodeConnectivity) {
        cellFaceNodes.push_back(getFacesForCell(conn));
    }
}

std::vector<std::vector<std::size_t>> PolyMesh::getFacesForCell(
    const std::vector<std::size_t>& conn) const {

    std::vector<std::vector<std::size_t>> faces;
    std::size_t numNodes = conn.size();

    if (dimension == 2) {
        // For 2D cells, faces are edges
        faces.reserve(numNodes);
        for (std::size_t i = 0; i < numNodes; ++i) {
            faces.push_back({conn[i], conn[(i + 1) % numNodes]});
        }
    } else if (dimension == 3) {
        // Templates for 3D cell face ordering
        static const std::vector<std::vector<std::size_t>> tetFaces = {
            {0, 1, 2}, {0, 3, 1}, {1, 3, 2}, {2, 0, 3}
        };
        static const std::vector<std::vector<std::size_t>> hexFaces = {
            {0, 1, 2, 3}, {4, 5, 6, 7}, {0, 1, 5, 4},
            {1, 2, 6, 5}, {2, 3, 7, 6}, {3, 0, 4, 7}
        };
        static const std::vector<std::vector<std::size_t>> wedgeFaces = {
            {0, 1, 2}, {3, 4, 5}, {0, 1, 4, 3}, {1, 2, 5, 4}, {2, 0, 3, 5}
        };

        const std::vector<std::vector<std::size_t>>* faceTemplate = nullptr;

        if (numNodes == 4) {
            faceTemplate = &tetFaces;
        } else if (numNodes == 8) {
            faceTemplate = &hexFaces;
        } else if (numNodes == 6) {
            faceTemplate = &wedgeFaces;
        }

        if (faceTemplate) {
            faces.reserve(faceTemplate->size());
            for (const auto& face : *faceTemplate) {
                std::vector<std::size_t> faceNodes;
                faceNodes.reserve(face.size());
                for (std::size_t idx : face) {
                    faceNodes.push_back(conn[idx]);
                }
                faces.push_back(std::move(faceNodes));
            }
        }
    }

    return faces;
}

void PolyMesh::computeFaceTopology() {
    if (cellFaceNodes.empty()) return;

    // Build lookup map: face (sorted node tuple) -> cells sharing it
    std::map<std::vector<std::size_t>, std::vector<std::size_t>> faceToCellMap;

    for (std::size_t cellIdx = 0; cellIdx < nCells; ++cellIdx) {
        for (const auto& face : cellFaceNodes[cellIdx]) {
            std::vector<std::size_t> key = face;
            std::sort(key.begin(), key.end());
            faceToCellMap[key].push_back(cellIdx);
        }
    }

    // Build boundary face lookup
    std::map<std::set<std::size_t>, int> boundaryFaceMap;
    for (std::size_t i = 0; i < boundaryFaceNodes.size(); ++i) {
        std::set<std::size_t> key(boundaryFaceNodes[i].begin(),
                                   boundaryFaceNodes[i].end());
        boundaryFaceMap[key] = boundaryFaceTags[i];
    }

    // Initialize arrays
    std::size_t maxFaces = 0;
    for (const auto& faces : cellFaceNodes) {
        maxFaces = std::max(maxFaces, faces.size());
    }

    cellNeighbors.resize(nCells);
    cellFaceTags.resize(nCells);

    for (std::size_t cellIdx = 0; cellIdx < nCells; ++cellIdx) {
        std::size_t numFaces = cellFaceNodes[cellIdx].size();
        cellNeighbors[cellIdx].resize(numFaces, -1);
        cellFaceTags[cellIdx].resize(numFaces, 0);

        for (std::size_t faceIdx = 0; faceIdx < numFaces; ++faceIdx) {
            std::vector<std::size_t> key = cellFaceNodes[cellIdx][faceIdx];
            std::sort(key.begin(), key.end());

            const auto& sharedCells = faceToCellMap[key];

            if (sharedCells.size() == 2) {
                // Internal face
                std::size_t neighborIdx = (sharedCells[0] == cellIdx)
                                          ? sharedCells[1]
                                          : sharedCells[0];
                cellNeighbors[cellIdx][faceIdx] = static_cast<int>(neighborIdx);
            } else if (sharedCells.size() == 1) {
                // Boundary face
                std::set<std::size_t> faceKeySet(key.begin(), key.end());
                auto it = boundaryFaceMap.find(faceKeySet);
                if (it != boundaryFaceMap.end()) {
                    cellFaceTags[cellIdx][faceIdx] = it->second;
                }
            }
        }
    }
}

// =========================================================================
// Geometric Property Computations
// =========================================================================

void PolyMesh::computeCellCentroids() {
    cellCentroids.resize(nCells);

    for (std::size_t i = 0; i < nCells; ++i) {
        const auto& conn = cellNodeConnectivity[i];
        std::array<double, 3> centroid = {0.0, 0.0, 0.0};

        for (std::size_t nodeIdx : conn) {
            centroid[0] += nodeCoords[nodeIdx][0];
            centroid[1] += nodeCoords[nodeIdx][1];
            centroid[2] += nodeCoords[nodeIdx][2];
        }

        double n = static_cast<double>(conn.size());
        centroid[0] /= n;
        centroid[1] /= n;
        centroid[2] /= n;

        cellCentroids[i] = centroid;
    }
}

void PolyMesh::computeFaceProperties() {
    if (cellNeighbors.empty()) return;

    cellFaceMidpoints.resize(nCells);
    cellFaceNormals.resize(nCells);
    cellFaceAreas.resize(nCells);

    for (std::size_t ci = 0; ci < nCells; ++ci) {
        std::size_t numFaces = cellFaceNodes[ci].size();
        cellFaceMidpoints[ci].resize(numFaces);
        cellFaceNormals[ci].resize(numFaces);
        cellFaceAreas[ci].resize(numFaces);

        for (std::size_t fi = 0; fi < numFaces; ++fi) {
            const auto& faceNodeIndices = cellFaceNodes[ci][fi];
            std::vector<std::array<double, 3>> nodes;
            nodes.reserve(faceNodeIndices.size());

            for (std::size_t nodeIdx : faceNodeIndices) {
                nodes.push_back(nodeCoords[nodeIdx]);
            }

            // Compute midpoint
            std::array<double, 3> midpoint = {0.0, 0.0, 0.0};
            for (const auto& node : nodes) {
                midpoint[0] += node[0];
                midpoint[1] += node[1];
                midpoint[2] += node[2];
            }
            double n = static_cast<double>(nodes.size());
            midpoint[0] /= n;
            midpoint[1] /= n;
            midpoint[2] /= n;
            cellFaceMidpoints[ci][fi] = midpoint;

            if (dimension == 2) {
                compute2DFaceMetrics(ci, fi, nodes);
            } else if (dimension == 3) {
                compute3DFaceMetrics(ci, fi, nodes);
            }
        }
    }

    orientFaceNormals();
}

void PolyMesh::compute2DFaceMetrics(std::size_t ci, std::size_t fi,
                                     const std::vector<std::array<double, 3>>& nodes) {
    // Edge vector
    double edgeVec[3] = {
        nodes[1][0] - nodes[0][0],
        nodes[1][1] - nodes[0][1],
        nodes[1][2] - nodes[0][2]
    };

    double length = std::sqrt(edgeVec[0] * edgeVec[0] +
                              edgeVec[1] * edgeVec[1] +
                              edgeVec[2] * edgeVec[2]);

    cellFaceAreas[ci][fi] = length;

    if (length > 1e-12) {
        // Perpendicular vector in 2D plane
        cellFaceNormals[ci][fi] = {edgeVec[1] / length, -edgeVec[0] / length, 0.0};
    } else {
        cellFaceNormals[ci][fi] = {0.0, 0.0, 0.0};
    }
}

void PolyMesh::compute3DFaceMetrics(std::size_t ci, std::size_t fi,
                                     const std::vector<std::array<double, 3>>& nodes) {
    // Compute centroid
    std::array<double, 3> centroid = {0.0, 0.0, 0.0};
    for (const auto& node : nodes) {
        centroid[0] += node[0];
        centroid[1] += node[1];
        centroid[2] += node[2];
    }
    double n = static_cast<double>(nodes.size());
    centroid[0] /= n;
    centroid[1] /= n;
    centroid[2] /= n;

    // Sum of cross products
    std::array<double, 3> areaVec = {0.0, 0.0, 0.0};
    std::size_t numNodes = nodes.size();

    for (std::size_t k = 0; k < numNodes; ++k) {
        const auto& p1 = nodes[k];
        const auto& p2 = nodes[(k + 1) % numNodes];

        double v1[3] = {p1[0] - centroid[0], p1[1] - centroid[1], p1[2] - centroid[2]};
        double v2[3] = {p2[0] - centroid[0], p2[1] - centroid[1], p2[2] - centroid[2]};

        // Cross product
        areaVec[0] += v1[1] * v2[2] - v1[2] * v2[1];
        areaVec[1] += v1[2] * v2[0] - v1[0] * v2[2];
        areaVec[2] += v1[0] * v2[1] - v1[1] * v2[0];
    }

    areaVec[0] /= 2.0;
    areaVec[1] /= 2.0;
    areaVec[2] /= 2.0;

    double area = std::sqrt(areaVec[0] * areaVec[0] +
                            areaVec[1] * areaVec[1] +
                            areaVec[2] * areaVec[2]);

    cellFaceAreas[ci][fi] = area;

    if (area > 1e-12) {
        cellFaceNormals[ci][fi] = {areaVec[0] / area,
                                    areaVec[1] / area,
                                    areaVec[2] / area};
    } else {
        cellFaceNormals[ci][fi] = {0.0, 0.0, 0.0};
    }
}

void PolyMesh::orientFaceNormals() {
    for (std::size_t ci = 0; ci < nCells; ++ci) {
        for (std::size_t fi = 0; fi < cellFaceNodes[ci].size(); ++fi) {
            // Vector from cell centroid to face midpoint
            double vecToFace[3] = {
                cellFaceMidpoints[ci][fi][0] - cellCentroids[ci][0],
                cellFaceMidpoints[ci][fi][1] - cellCentroids[ci][1],
                cellFaceMidpoints[ci][fi][2] - cellCentroids[ci][2]
            };

            // Dot product with normal
            double dot = vecToFace[0] * cellFaceNormals[ci][fi][0] +
                         vecToFace[1] * cellFaceNormals[ci][fi][1] +
                         vecToFace[2] * cellFaceNormals[ci][fi][2];

            if (dot < 0) {
                // Flip the normal
                cellFaceNormals[ci][fi][0] *= -1;
                cellFaceNormals[ci][fi][1] *= -1;
                cellFaceNormals[ci][fi][2] *= -1;
            }
        }
    }
}

void PolyMesh::computeFaceToCentroidDist() {
    if (cellFaceMidpoints.empty()) return;

    faceToCentroidDistances.resize(nCells);

    for (std::size_t ci = 0; ci < nCells; ++ci) {
        std::size_t numFaces = cellFaceNodes[ci].size();
        faceToCentroidDistances[ci].resize(numFaces);

        for (std::size_t fi = 0; fi < numFaces; ++fi) {
            double vec[3] = {
                cellFaceMidpoints[ci][fi][0] - cellCentroids[ci][0],
                cellFaceMidpoints[ci][fi][1] - cellCentroids[ci][1],
                cellFaceMidpoints[ci][fi][2] - cellCentroids[ci][2]
            };

            faceToCentroidDistances[ci][fi] = std::sqrt(
                vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
        }
    }
}

void PolyMesh::computeCellVolumes() {
    if (dimension == 2) {
        compute2DCellVolumes();
    } else if (dimension == 3) {
        compute3DCellVolumes();
    } else {
        cellVolumes.resize(nCells, 0.0);
    }
}

void PolyMesh::compute2DCellVolumes() {
    cellVolumes.resize(nCells);

    for (std::size_t i = 0; i < nCells; ++i) {
        const auto& conn = cellNodeConnectivity[i];
        std::size_t n = conn.size();

        // Shoelace formula for polygon area
        double area = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            const auto& p1 = nodeCoords[conn[j]];
            const auto& p2 = nodeCoords[conn[(j + 1) % n]];
            area += p1[0] * p2[1] - p2[0] * p1[1];
        }
        cellVolumes[i] = std::abs(area) / 2.0;
    }
}

void PolyMesh::compute3DCellVolumes() {
    cellVolumes.resize(nCells);

    // Using divergence theorem: V = (1/3) * sum(face_midpoint . face_normal * face_area)
    for (std::size_t ci = 0; ci < nCells; ++ci) {
        double volume = 0.0;
        std::size_t numFaces = cellFaceNodes[ci].size();

        for (std::size_t fi = 0; fi < numFaces; ++fi) {
            double dot = cellFaceMidpoints[ci][fi][0] * cellFaceNormals[ci][fi][0] +
                         cellFaceMidpoints[ci][fi][1] * cellFaceNormals[ci][fi][1] +
                         cellFaceMidpoints[ci][fi][2] * cellFaceNormals[ci][fi][2];
            volume += dot * cellFaceAreas[ci][fi];
        }

        cellVolumes[ci] = volume / 3.0;
    }
}

// =========================================================================
// Helper Methods
// =========================================================================

int PolyMesh::getMeshDimension(const std::vector<int>& elemTypes) const {
    int maxDim = 0;

    for (int et : elemTypes) {
        std::string name;
        int dim, order, numNodes, numPrimaryNodes;
        std::vector<double> localNodeCoords;
        gmsh::model::mesh::getElementProperties(et, name, dim, order, numNodes,
                                                 localNodeCoords, numPrimaryNodes);
        maxDim = std::max(maxDim, dim);
    }

    return maxDim;
}

void PolyMesh::printGeneralInfo() const {
    std::cout << "\n" << std::setw(40) << "--- General Information ---" << "\n\n";
    std::cout << "  " << std::left << std::setw(25) << "Dimension:" << dimension << "D\n";
    std::cout << "  " << std::left << std::setw(25) << "Number of Nodes:" << nNodes << "\n";
    std::cout << "  " << std::left << std::setw(25) << "Number of Cells:" << nCells << "\n";
}

void PolyMesh::printGeometricProperties() const {
    if (nNodes == 0) return;

    double minX = nodeCoords[0][0], maxX = nodeCoords[0][0];
    double minY = nodeCoords[0][1], maxY = nodeCoords[0][1];
    double minZ = nodeCoords[0][2], maxZ = nodeCoords[0][2];

    for (const auto& coord : nodeCoords) {
        minX = std::min(minX, coord[0]);
        maxX = std::max(maxX, coord[0]);
        minY = std::min(minY, coord[1]);
        maxY = std::max(maxY, coord[1]);
        minZ = std::min(minZ, coord[2]);
        maxZ = std::max(maxZ, coord[2]);
    }

    std::cout << "\n" << std::setw(40) << "--- Geometric Bounding Box ---" << "\n\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  " << std::left << std::setw(25) << "X Range:"
              << minX << " to " << maxX << "\n";
    std::cout << "  " << std::left << std::setw(25) << "Y Range:"
              << minY << " to " << maxY << "\n";
    if (dimension == 3) {
        std::cout << "  " << std::left << std::setw(25) << "Z Range:"
                  << minZ << " to " << maxZ << "\n";
    }
}

void PolyMesh::printCellGeometry() const {
    if (cellVolumes.empty()) return;

    std::cout << "\n" << std::setw(40) << "--- Cell Geometry ---" << "\n\n";
    printCellTypeDistribution();

    std::cout << "\n  " << std::left << std::setw(20) << "Metric"
              << std::right << std::setw(15) << "Min"
              << std::setw(15) << "Max"
              << std::setw(15) << "Average" << "\n";
    std::cout << "  " << std::string(19, '-') << " "
              << std::string(15, '-') << " "
              << std::string(15, '-') << " "
              << std::string(15, '-') << "\n";

    printStatLine("Cell Volume", cellVolumes);

    // Flatten face-to-centroid distances
    std::vector<double> allDistances;
    for (const auto& dists : faceToCentroidDistances) {
        for (double d : dists) {
            if (d > 0) allDistances.push_back(d);
        }
    }
    printStatLine("Face-to-Centroid Dist", allDistances);
}

void PolyMesh::printCellTypeDistribution() const {
    if (cellElementTypes.empty()) return;

    std::map<int, std::size_t> typeCounts;
    for (int type : cellElementTypes) {
        typeCounts[type]++;
    }

    std::cout << "  Cell Type Distribution:\n";
    for (const auto& [typeId, count] : typeCounts) {
        std::string typeName = "Unknown";
        auto it = elementTypeProperties.find(typeId);
        if (it != elementTypeProperties.end()) {
            typeName = it->second.name;
        }
        std::cout << "    - " << std::left << std::setw(20)
                  << (typeName + ":") << count << "\n";
    }
}

void PolyMesh::printStatLine(const std::string& name,
                              const std::vector<double>& data) const {
    if (data.empty()) return;

    std::vector<double> validData;
    for (double v : data) {
        if (v > 0) validData.push_back(v);
    }
    if (validData.empty()) return;

    double minVal = *std::min_element(validData.begin(), validData.end());
    double maxVal = *std::max_element(validData.begin(), validData.end());
    double avgVal = std::accumulate(validData.begin(), validData.end(), 0.0) /
                    static_cast<double>(validData.size());

    std::cout << std::scientific << std::setprecision(4);
    std::cout << "  " << std::left << std::setw(25) << name
              << std::right << std::setw(15) << minVal
              << std::setw(15) << maxVal
              << std::setw(15) << avgVal << "\n";
}

}  // namespace fvm
