#pragma once

/**
 * @file mesh_partition_manager.hpp
 * @brief Manager for partitioning meshes and creating local mesh instances.
 *
 * This module provides the MeshPartitionManager class, which handles the
 * logic of taking a global PolyMesh, partitioning it, computing halo cell
 * information, and creating multiple LocalMesh instances for distributed
 * environments.
 */

#include "local_mesh.hpp"
#include "partition.hpp"

#include <map>
#include <string>
#include <vector>

namespace fvm {

/**
 * @brief Manages the partitioning of a global mesh and creation of local meshes.
 *
 * This class provides static methods to perform partitioning and mesh creation
 * tasks. It handles:
 * - Partitioning the global mesh using METIS or hierarchical bisection
 * - Computing halo cell information for inter-partition communication
 * - Creating LocalMesh instances for each partition
 */
class FVM_API MeshPartitionManager {
public:
    /**
     * @brief Partitions a global mesh and creates a list of local mesh objects.
     *
     * @param globalMesh The complete, unpartitioned PolyMesh object (must be analyzed)
     * @param nParts The desired number of partitions
     * @param partitionMethod The algorithm to use: "metis" or "hierarchical"
     * @param cellReorderStrategy Optional strategy to reorder cells within each
     *                            local mesh (e.g., "rcm", "gps"). Empty for no reordering.
     * @param nodeReorderStrategy Optional strategy to reorder nodes within each
     *                            local mesh (e.g., "rcm"). Empty for no reordering.
     * @param storeOrder If true, stores original ordering for later restoration
     * @return A vector of LocalMesh objects, one for each partition
     * @throws std::runtime_error if partitioning fails
     */
    static std::vector<LocalMesh> createLocalMeshes(
        const PolyMesh& globalMesh,
        int nParts,
        const std::string& partitionMethod = "metis",
        const std::string& cellReorderStrategy = "",
        const std::string& nodeReorderStrategy = "",
        bool storeOrder = false
    );

    /**
     * @brief Creates local meshes from pre-computed partition assignments.
     *
     * @param globalMesh The complete, unpartitioned PolyMesh object
     * @param cellPartitions Partition ID for each cell (size = nCells)
     * @param cellReorderStrategy Optional cell reordering strategy
     * @param nodeReorderStrategy Optional node reordering strategy
     * @param storeOrder If true, stores original ordering for restoration
     * @return A vector of LocalMesh objects
     */
    static std::vector<LocalMesh> createLocalMeshesFromPartitions(
        const PolyMesh& globalMesh,
        const std::vector<int>& cellPartitions,
        const std::string& cellReorderStrategy = "",
        const std::string& nodeReorderStrategy = "",
        bool storeOrder = false
    );

private:
    /// Type for send candidate map: {sender_rank: {receiver_rank: [global_cell_indices]}}
    using SendCandidateMap = std::map<int, std::map<int, std::vector<std::size_t>>>;

    /**
     * @brief Determines which cells each partition needs to send to neighbors.
     *
     * Identifies cells adjacent to cells in a different partition, which are
     * candidates for halo exchange.
     *
     * @param globalMesh The global PolyMesh object
     * @param cellPartitions Partition ID for each cell
     * @return A map of send candidates for each partition
     */
    static SendCandidateMap findSendCandidates(
        const PolyMesh& globalMesh,
        const std::vector<int>& cellPartitions
    );

    /**
     * @brief Computes halo information for all partitions.
     *
     * Builds owned cells, halo cells, and send/recv maps for each partition.
     *
     * @param globalMesh The global PolyMesh object
     * @param cellPartitions Partition ID for each cell
     * @param sendCandidates Send candidate map from findSendCandidates
     * @return A map from rank to HaloInfo
     */
    static std::map<int, HaloInfo> computeHaloIndices(
        const PolyMesh& globalMesh,
        const std::vector<int>& cellPartitions,
        const SendCandidateMap& sendCandidates
    );
};

}  // namespace fvm
