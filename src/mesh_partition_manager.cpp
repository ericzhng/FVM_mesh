#include "mesh_partition_manager.hpp"

#include <algorithm>
#include <iostream>
#include <set>
#include <stdexcept>

namespace fvm {

// =========================================================================
// Public API
// =========================================================================

std::vector<LocalMesh> MeshPartitionManager::createLocalMeshes(
    const PolyMesh& globalMesh,
    int nParts,
    const std::string& partitionMethod,
    const std::string& cellReorderStrategy,
    const std::string& nodeReorderStrategy,
    bool storeOrder
) {
    if (nParts <= 0) {
        throw std::invalid_argument("nParts must be a positive integer.");
    }

    // Ensure mesh is analyzed
    if (!globalMesh.isAnalyzed()) {
        throw std::runtime_error("Global mesh must be analyzed before partitioning.");
    }

    // Partition the mesh
    std::vector<int> cellPartitions = partitionMesh(globalMesh, nParts, partitionMethod);
    printPartitionSummary(cellPartitions);

    return createLocalMeshesFromPartitions(
        globalMesh,
        cellPartitions,
        cellReorderStrategy,
        nodeReorderStrategy,
        storeOrder
    );
}

std::vector<LocalMesh> MeshPartitionManager::createLocalMeshesFromPartitions(
    const PolyMesh& globalMesh,
    const std::vector<int>& cellPartitions,
    const std::string& cellReorderStrategy,
    const std::string& nodeReorderStrategy,
    bool storeOrder
) {
    if (cellPartitions.empty()) {
        return {};
    }

    // Determine number of partitions
    int nParts = *std::max_element(cellPartitions.begin(), cellPartitions.end()) + 1;

    if (nParts == 0) {
        return {};
    }

    // Compute halo information
    SendCandidateMap sendCandidates = findSendCandidates(globalMesh, cellPartitions);
    std::map<int, HaloInfo> haloIndices = computeHaloIndices(
        globalMesh, cellPartitions, sendCandidates
    );

    // Create local meshes
    std::vector<LocalMesh> localMeshes;
    localMeshes.reserve(nParts);

    for (int rank = 0; rank < nParts; ++rank) {
        auto it = haloIndices.find(rank);
        if (it == haloIndices.end() || it->second.ownedCells.empty()) {
            // Skip empty partitions
            continue;
        }

        // Determine if we need to store order for reordering
        bool needStoreOrder = storeOrder ||
                              !cellReorderStrategy.empty() ||
                              !nodeReorderStrategy.empty();

        LocalMesh localMesh = LocalMesh::fromGlobalMesh(
            globalMesh,
            it->second,
            rank,
            needStoreOrder
        );

        // Apply reordering if requested
        if (!cellReorderStrategy.empty()) {
            localMesh.reorderCells(cellReorderStrategy);
        }
        if (!nodeReorderStrategy.empty()) {
            localMesh.reorderNodes(nodeReorderStrategy);
        }

        // Ensure mesh is analyzed
        localMesh.analyzeMesh();

        localMeshes.push_back(std::move(localMesh));
    }

    return localMeshes;
}

// =========================================================================
// Private Methods
// =========================================================================

MeshPartitionManager::SendCandidateMap MeshPartitionManager::findSendCandidates(
    const PolyMesh& globalMesh,
    const std::vector<int>& cellPartitions
) {
    if (cellPartitions.empty()) {
        return {};
    }

    int nParts = *std::max_element(cellPartitions.begin(), cellPartitions.end()) + 1;

    // Initialize send candidates using sets for deduplication
    std::map<int, std::map<int, std::set<std::size_t>>> sendCandidateSets;
    for (int r = 0; r < nParts; ++r) {
        sendCandidateSets[r] = {};
    }

    // For each cell, check if any neighbor is in a different partition
    for (std::size_t gIdx = 0; gIdx < globalMesh.nCells; ++gIdx) {
        int ownerPart = cellPartitions[gIdx];

        for (int neighborGIdx : globalMesh.cellNeighbors[gIdx]) {
            if (neighborGIdx >= 0 &&
                neighborGIdx < static_cast<int>(cellPartitions.size())) {

                int neighborPart = cellPartitions[neighborGIdx];

                if (ownerPart != neighborPart) {
                    // This cell needs to be sent to the neighbor's partition
                    sendCandidateSets[ownerPart][neighborPart].insert(gIdx);
                }
            }
        }
    }

    // Convert sets to sorted vectors
    SendCandidateMap sendCandidates;
    for (int r = 0; r < nParts; ++r) {
        sendCandidates[r] = {};
        for (const auto& [neighborRank, cellSet] : sendCandidateSets[r]) {
            std::vector<std::size_t> cells(cellSet.begin(), cellSet.end());
            std::sort(cells.begin(), cells.end());
            sendCandidates[r][neighborRank] = std::move(cells);
        }
    }

    return sendCandidates;
}

std::map<int, HaloInfo> MeshPartitionManager::computeHaloIndices(
    const PolyMesh& globalMesh,
    const std::vector<int>& cellPartitions,
    const SendCandidateMap& sendCandidates
) {
    if (cellPartitions.empty()) {
        return {};
    }

    int nParts = *std::max_element(cellPartitions.begin(), cellPartitions.end()) + 1;

    // Handle single partition case
    if (nParts <= 1) {
        if (globalMesh.nCells > 0) {
            HaloInfo info;
            info.ownedCells.reserve(globalMesh.nCells);
            for (std::size_t i = 0; i < globalMesh.nCells; ++i) {
                info.ownedCells.push_back(i);
            }
            return {{0, std::move(info)}};
        }
        return {};
    }

    std::map<int, HaloInfo> haloData;

    for (int rank = 0; rank < nParts; ++rank) {
        HaloInfo info;

        // Find owned cells (cells belonging to this partition)
        for (std::size_t gIdx = 0; gIdx < globalMesh.nCells; ++gIdx) {
            if (cellPartitions[gIdx] == rank) {
                info.ownedCells.push_back(gIdx);
            }
        }

        if (info.ownedCells.empty()) {
            continue;  // Skip empty partitions
        }

        // Build global-to-local map for owned cells
        std::unordered_map<std::size_t, std::size_t> g2lOwned;
        for (std::size_t l = 0; l < info.ownedCells.size(); ++l) {
            g2lOwned[info.ownedCells[l]] = l;
        }

        // Collect halo cells this rank will receive
        // A halo cell is a cell owned by another partition that will be sent to us
        std::vector<std::size_t> haloCellsG;
        std::map<int, std::vector<std::size_t>> haloFromNeighborsG;

        for (const auto& [senderRank, sendDict] : sendCandidates) {
            auto it = sendDict.find(rank);
            if (it != sendDict.end()) {
                // senderRank is sending cells to us
                const auto& cellsToRecv = it->second;
                haloFromNeighborsG[senderRank] = cellsToRecv;
                haloCellsG.insert(haloCellsG.end(), cellsToRecv.begin(), cellsToRecv.end());
            }
        }

        info.haloCells = std::move(haloCellsG);

        // Build global-to-local map for halo cells
        // Halo cells start after owned cells
        std::unordered_map<std::size_t, std::size_t> g2lHalo;
        for (std::size_t l = 0; l < info.haloCells.size(); ++l) {
            g2lHalo[info.haloCells[l]] = l + info.ownedCells.size();
        }

        // Build local send map
        auto sendIt = sendCandidates.find(rank);
        if (sendIt != sendCandidates.end()) {
            for (const auto& [neighborRank, gIndicesToSend] : sendIt->second) {
                std::vector<std::size_t> localIndicesToSend;
                localIndicesToSend.reserve(gIndicesToSend.size());

                for (std::size_t g : gIndicesToSend) {
                    auto localIt = g2lOwned.find(g);
                    if (localIt != g2lOwned.end()) {
                        localIndicesToSend.push_back(localIt->second);
                    }
                }

                info.sendMap[neighborRank] = std::move(localIndicesToSend);
            }
        }

        // Build local recv map
        for (const auto& [neighborRank, gIndicesToRecv] : haloFromNeighborsG) {
            std::vector<std::size_t> localIndicesToRecv;
            localIndicesToRecv.reserve(gIndicesToRecv.size());

            for (std::size_t g : gIndicesToRecv) {
                auto localIt = g2lHalo.find(g);
                if (localIt != g2lHalo.end()) {
                    localIndicesToRecv.push_back(localIt->second);
                }
            }

            info.recvMap[neighborRank] = std::move(localIndicesToRecv);
        }

        haloData[rank] = std::move(info);
    }

    return haloData;
}

}  // namespace fvm
