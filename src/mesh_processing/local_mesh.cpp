#include "local_mesh.hpp"
#include "reorder.hpp"

#include <algorithm>
#include <numeric>
#include <queue>
#include <set>
#include <stdexcept>

namespace fvm {

// =========================================================================
// Factory Method
// =========================================================================

LocalMesh LocalMesh::fromGlobalMesh(
    const PolyMesh& globalMesh,
    const HaloInfo& haloInfo,
    int rank,
    bool storeOrder
) {
    if (!globalMesh.isAnalyzed()) {
        throw std::runtime_error("Mesh must be analyzed before creating a LocalMesh.");
    }

    if (haloInfo.ownedCells.empty()) {
        throw std::runtime_error("HaloInfo must contain at least one owned cell.");
    }

    if (rank < 0) {
        throw std::runtime_error("Rank must be a non-negative integer.");
    }

    LocalMesh localMesh;
    localMesh.rank = rank;
    localMesh.numOwnedCells = haloInfo.ownedCells.size();
    localMesh.numHaloCells = haloInfo.haloCells.size();
    localMesh.sendMap = haloInfo.sendMap;
    localMesh.recvMap = haloInfo.recvMap;

    // Initialize local-to-global cell mapping
    // Owned cells come first, followed by halo cells
    localMesh.l2gCells.reserve(localMesh.numOwnedCells + localMesh.numHaloCells);
    for (std::size_t g : haloInfo.ownedCells) {
        localMesh.l2gCells.push_back(g);
    }
    for (std::size_t g : haloInfo.haloCells) {
        localMesh.l2gCells.push_back(g);
    }

    // Build global-to-local cell mapping
    for (std::size_t l = 0; l < localMesh.l2gCells.size(); ++l) {
        localMesh.g2lCells[localMesh.l2gCells[l]] = l;
    }

    // Initialize node mappings
    // Collect all unique nodes from local cells
    std::set<std::size_t> uniqueNodesG;
    for (std::size_t g : localMesh.l2gCells) {
        for (std::size_t nodeG : globalMesh.cellNodeConnectivity[g]) {
            uniqueNodesG.insert(nodeG);
        }
    }

    localMesh.l2gNodes.assign(uniqueNodesG.begin(), uniqueNodesG.end());
    for (std::size_t l = 0; l < localMesh.l2gNodes.size(); ++l) {
        localMesh.g2lNodes[localMesh.l2gNodes[l]] = l;
    }

    // Build local mesh data from global mesh
    localMesh.buildFromGlobalMesh(globalMesh);

    // Store original ordering if requested
    if (storeOrder) {
        localMesh.storeOriginalOrdering();
    }

    return localMesh;
}

// =========================================================================
// Internal Methods
// =========================================================================

void LocalMesh::buildFromGlobalMesh(const PolyMesh& globalMesh) {
    // Set basic properties
    dimension = globalMesh.dimension;
    nCells = l2gCells.size();
    nNodes = l2gNodes.size();

    // Copy node coordinates using local-to-global mapping
    nodeCoords.resize(nNodes);
    for (std::size_t l = 0; l < nNodes; ++l) {
        nodeCoords[l] = globalMesh.nodeCoords[l2gNodes[l]];
    }

    // Remap cell connectivity from global to local node indices
    cellNodeConnectivity.resize(nCells);
    for (std::size_t l = 0; l < nCells; ++l) {
        std::size_t g = l2gCells[l];
        const auto& globalConn = globalMesh.cellNodeConnectivity[g];

        cellNodeConnectivity[l].reserve(globalConn.size());
        for (std::size_t nodeG : globalConn) {
            auto it = g2lNodes.find(nodeG);
            if (it == g2lNodes.end()) {
                throw std::runtime_error(
                    "Global node index " + std::to_string(nodeG) +
                    " not found in local node mapping."
                );
            }
            cellNodeConnectivity[l].push_back(it->second);
        }
    }

    // Copy element types
    if (!globalMesh.cellElementTypes.empty()) {
        cellElementTypes.resize(nCells);
        for (std::size_t l = 0; l < nCells; ++l) {
            cellElementTypes[l] = globalMesh.cellElementTypes[l2gCells[l]];
        }
    }

    // Copy element type properties
    elementTypeProperties = globalMesh.elementTypeProperties;

    // Populate boundary face information
    populateBoundaryFaces(globalMesh);
}

void LocalMesh::populateBoundaryFaces(const PolyMesh& globalMesh) {
    std::vector<std::vector<std::size_t>> localBoundaryFaceNodes;
    std::vector<int> localBoundaryFaceTags;

    // Iterate through global boundary faces
    for (std::size_t i = 0; i < globalMesh.boundaryFaceNodes.size(); ++i) {
        const auto& faceNodesG = globalMesh.boundaryFaceNodes[i];

        // Check if all nodes of this face are in our local mesh
        bool allNodesLocal = true;
        for (std::size_t nodeG : faceNodesG) {
            if (g2lNodes.find(nodeG) == g2lNodes.end()) {
                allNodesLocal = false;
                break;
            }
        }

        if (allNodesLocal) {
            // Remap face nodes to local indices
            std::vector<std::size_t> faceNodesL;
            faceNodesL.reserve(faceNodesG.size());
            for (std::size_t nodeG : faceNodesG) {
                faceNodesL.push_back(g2lNodes[nodeG]);
            }

            localBoundaryFaceNodes.push_back(std::move(faceNodesL));
            localBoundaryFaceTags.push_back(globalMesh.boundaryFaceTags[i]);
        }
    }

    boundaryFaceNodes = std::move(localBoundaryFaceNodes);
    boundaryFaceTags = std::move(localBoundaryFaceTags);
    boundaryPatchMap = globalMesh.boundaryPatchMap;
}

void LocalMesh::storeOriginalOrdering() {
    originalL2gCells_ = l2gCells;
    originalG2lCells_ = g2lCells;
    originalL2gNodes_ = l2gNodes;
    originalG2lNodes_ = g2lNodes;
    originalNodeCoords_ = nodeCoords;
    originalCellNodeConnectivity_ = cellNodeConnectivity;
    originalCellElementTypes_ = cellElementTypes;
    originalSendMap_ = sendMap;
    originalRecvMap_ = recvMap;
    hasStoredOrder_ = true;
}

void LocalMesh::restoreOriginalCellOrdering() {
    if (!hasStoredOrder_) return;

    l2gCells = originalL2gCells_;
    g2lCells = originalG2lCells_;
    cellNodeConnectivity = originalCellNodeConnectivity_;
    cellElementTypes = originalCellElementTypes_;
    sendMap = originalSendMap_;
    recvMap = originalRecvMap_;
    useReorderedCells = false;
}

void LocalMesh::restoreOriginalNodeOrdering() {
    if (!hasStoredOrder_) return;

    l2gNodes = originalL2gNodes_;
    g2lNodes = originalG2lNodes_;
    nodeCoords = originalNodeCoords_;
    cellNodeConnectivity = originalCellNodeConnectivity_;
    useReorderedNodes = false;
}

// =========================================================================
// Reordering Methods
// =========================================================================

void LocalMesh::reorderCells(const std::string& strategy, bool restore) {
    if (restore) {
        restoreOriginalCellOrdering();
        analyzeMesh();
        return;
    }

    // Store old l2g mapping for remapping communication maps
    std::vector<std::size_t> oldL2gCells = l2gCells;

    // Reorder cells (only owned cells, halo stays at end)
    renumberCells(*this, strategy, numOwnedCells);

    // Update l2g mapping - we need to recompute based on new order
    // The renumberCells function reordered cellNodeConnectivity
    // We need to track which original cells are now where

    // Build mapping from old local index to new local index
    // Since renumberCells uses getOrder which returns new->old mapping,
    // after the reorder, cellNodeConnectivity[newIdx] has data from oldIdx
    // We need to update l2gCells accordingly

    // For now, we'll recompute l2g from the known fact that:
    // - owned cells are reordered within [0, numOwnedCells)
    // - halo cells remain in [numOwnedCells, nCells)

    // Create reorder strategy to get the order that was used
    auto strat = createCellReorderStrategy(strategy);

    // Create temporary mesh view for owned cells
    PolyMesh tempMesh;
    tempMesh.dimension = dimension;
    tempMesh.nNodes = nNodes;
    tempMesh.nCells = numOwnedCells;
    tempMesh.nodeCoords = nodeCoords;

    // Use original connectivity for owned cells
    tempMesh.cellNodeConnectivity.reserve(numOwnedCells);
    for (std::size_t i = 0; i < numOwnedCells; ++i) {
        tempMesh.cellNodeConnectivity.push_back(
            originalCellNodeConnectivity_.empty()
                ? cellNodeConnectivity[i]
                : originalCellNodeConnectivity_[i]
        );
    }
    tempMesh.analyzeMesh();

    if (!cellCentroids.empty() && cellCentroids.size() >= numOwnedCells) {
        tempMesh.cellCentroids.resize(numOwnedCells);
        for (std::size_t i = 0; i < numOwnedCells; ++i) {
            tempMesh.cellCentroids[i] = cellCentroids[i];
        }
    }

    auto newOrder = strat->getOrder(tempMesh);

    // newOrder[newIdx] = oldIdx within owned cells
    // Update l2gCells for owned cells
    std::vector<std::size_t> newL2gCells(nCells);
    for (std::size_t newIdx = 0; newIdx < numOwnedCells; ++newIdx) {
        std::size_t oldIdx = newOrder[newIdx];
        newL2gCells[newIdx] = oldL2gCells[oldIdx];
    }
    // Keep halo cells unchanged
    for (std::size_t i = numOwnedCells; i < nCells; ++i) {
        newL2gCells[i] = oldL2gCells[i];
    }

    l2gCells = std::move(newL2gCells);

    // Rebuild g2l mapping
    g2lCells.clear();
    for (std::size_t l = 0; l < l2gCells.size(); ++l) {
        g2lCells[l2gCells[l]] = l;
    }

    // Remap send/recv maps to use new local indices
    // Build old-to-new local index mapping
    std::unordered_map<std::size_t, std::size_t> oldToNew;
    for (std::size_t newIdx = 0; newIdx < numOwnedCells; ++newIdx) {
        std::size_t oldIdx = newOrder[newIdx];
        oldToNew[oldIdx] = newIdx;
    }
    for (std::size_t i = numOwnedCells; i < nCells; ++i) {
        oldToNew[i] = i;  // Halo cells unchanged
    }

    for (auto& [neighborRank, indices] : sendMap) {
        for (std::size_t& idx : indices) {
            idx = oldToNew[idx];
        }
    }
    for (auto& [neighborRank, indices] : recvMap) {
        for (std::size_t& idx : indices) {
            idx = oldToNew[idx];
        }
    }

    useReorderedCells = true;
    analyzeMesh();
}

void LocalMesh::reorderNodes(const std::string& strategy, bool restore) {
    if (restore) {
        restoreOriginalNodeOrdering();
        analyzeMesh();
        return;
    }

    // Store old l2g mapping
    std::vector<std::size_t> oldL2gNodes = l2gNodes;

    // Reorder nodes
    renumberNodes(*this, strategy);

    // After reordering, nodeCoords is reordered and cellNodeConnectivity
    // is updated to use new node indices

    // We need to update l2gNodes to match the new ordering
    // renumberNodes creates newOrder where nodeCoords[newIdx] = oldCoords[newOrder[newIdx]]
    // So l2gNodes[newIdx] should be oldL2gNodes[newOrder[newIdx]]

    // Since we don't have direct access to the order used, we'll infer from
    // the fact that node coordinates are now reordered

    // For simplicity, we'll recompute l2gNodes based on the strategy
    SparseMatrix adj = buildNodeAdjacency(*this);
    std::vector<std::size_t> newOrder;

    if (strategy == "rcm") {
        // Find starting node with minimum degree
        std::size_t startNode = 0;
        std::size_t minDegree = std::numeric_limits<std::size_t>::max();
        for (std::size_t i = 0; i < nNodes; ++i) {
            if (adj.degree(i) < minDegree) {
                minDegree = adj.degree(i);
                startNode = i;
            }
        }

        // BFS
        newOrder.reserve(nNodes);
        std::vector<bool> visited(nNodes, false);
        std::queue<std::size_t> queue;
        queue.push(startNode);
        visited[startNode] = true;

        while (newOrder.size() < nNodes) {
            while (!queue.empty()) {
                std::size_t current = queue.front();
                queue.pop();
                newOrder.push_back(current);

                auto [begin, end] = adj.neighbors(current);
                std::vector<std::size_t> neighbors(begin, end);

                std::sort(neighbors.begin(), neighbors.end(),
                          [&adj](std::size_t a, std::size_t b) {
                              return adj.degree(a) < adj.degree(b);
                          });

                for (std::size_t neighbor : neighbors) {
                    if (!visited[neighbor]) {
                        visited[neighbor] = true;
                        queue.push(neighbor);
                    }
                }
            }

            if (newOrder.size() < nNodes) {
                for (std::size_t i = 0; i < nNodes; ++i) {
                    if (!visited[i]) {
                        queue.push(i);
                        visited[i] = true;
                        break;
                    }
                }
            }
        }

        std::reverse(newOrder.begin(), newOrder.end());
    } else {
        // For other strategies, assume identity for now
        newOrder.resize(nNodes);
        std::iota(newOrder.begin(), newOrder.end(), 0);
    }

    // Update l2gNodes
    std::vector<std::size_t> newL2gNodes(nNodes);
    for (std::size_t newIdx = 0; newIdx < nNodes; ++newIdx) {
        newL2gNodes[newIdx] = oldL2gNodes[newOrder[newIdx]];
    }
    l2gNodes = std::move(newL2gNodes);

    // Rebuild g2l mapping
    g2lNodes.clear();
    for (std::size_t l = 0; l < l2gNodes.size(); ++l) {
        g2lNodes[l2gNodes[l]] = l;
    }

    useReorderedNodes = true;
    analyzeMesh();
}

}  // namespace fvm
