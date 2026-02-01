#pragma once

/**
 * @file local_mesh.hpp
 * @brief Partitioned mesh for distributed computing environments.
 *
 * This module provides the LocalMesh class to manage partitioned meshes for
 * parallel computations. A LocalMesh represents the portion of a global mesh
 * assigned to a single process, including both the cells owned by that process
 * and the necessary halo cells from neighboring partitions.
 */

#include "poly_mesh.hpp"

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace fvm {

/**
 * @brief Information about halo cells for a partition.
 *
 * Contains the cell indices and communication maps needed to create
 * a LocalMesh for a specific rank.
 */
struct HaloInfo {
    /// Global indices of cells owned by this partition
    std::vector<std::size_t> ownedCells;

    /// Global indices of halo cells (ghost cells from neighbors)
    std::vector<std::size_t> haloCells;

    /// Map of {neighbor_rank: [local_cell_indices_to_send]}
    std::map<int, std::vector<std::size_t>> sendMap;

    /// Map of {neighbor_rank: [local_cell_indices_to_receive]}
    std::map<int, std::vector<std::size_t>> recvMap;
};

/**
 * @brief Represents the mesh for a single partition in a distributed setup.
 *
 * This class holds the geometric and topological data for a subset of the
 * global mesh, including cells owned by the current process and "halo" cells
 * from neighboring processes required for communication.
 */
class FVM_API LocalMesh : public PolyMesh {
public:
    // =========================================================================
    // Partition-Specific Data
    // =========================================================================

    /// The rank (partition ID) this mesh belongs to
    int rank = -1;

    /// Number of cells owned by this partition (first numOwnedCells in arrays)
    std::size_t numOwnedCells = 0;

    /// Number of halo (ghost) cells
    std::size_t numHaloCells = 0;

    /// Map from local cell indices to global cell indices
    std::vector<std::size_t> l2gCells;

    /// Map from global cell indices to local cell indices
    std::unordered_map<std::size_t, std::size_t> g2lCells;

    /// Map from local node indices to global node indices
    std::vector<std::size_t> l2gNodes;

    /// Map from global node indices to local node indices
    std::unordered_map<std::size_t, std::size_t> g2lNodes;

    /// Halo communication map: cells to send to each neighbor rank
    std::map<int, std::vector<std::size_t>> sendMap;

    /// Halo communication map: halo cells to receive from each neighbor rank
    std::map<int, std::vector<std::size_t>> recvMap;

    /// Flag indicating if cells have been reordered
    bool useReorderedCells = false;

    /// Flag indicating if nodes have been reordered
    bool useReorderedNodes = false;

    // =========================================================================
    // Constructors
    // =========================================================================

    LocalMesh() = default;
    ~LocalMesh() override = default;

    // Move semantics
    LocalMesh(LocalMesh&&) = default;
    LocalMesh& operator=(LocalMesh&&) = default;

    // Prevent copy (large data)
    LocalMesh(const LocalMesh&) = delete;
    LocalMesh& operator=(const LocalMesh&) = delete;

    // =========================================================================
    // Factory Method
    // =========================================================================

    /**
     * @brief Constructs a LocalMesh for a specific partition.
     *
     * @param globalMesh The complete, unpartitioned mesh (must be analyzed)
     * @param haloInfo Halo information for this rank
     * @param rank The ID of the partition
     * @param storeOrder If true, stores original ordering for later restoration
     * @return A new LocalMesh instance for the specified rank
     * @throws std::runtime_error if global mesh has not been analyzed
     */
    static LocalMesh fromGlobalMesh(
        const PolyMesh& globalMesh,
        const HaloInfo& haloInfo,
        int rank,
        bool storeOrder = false
    );

    // =========================================================================
    // Reordering Methods
    // =========================================================================

    /**
     * @brief Reorders the cells of the mesh to improve locality.
     *
     * This is a stateful operation that modifies the mesh in-place.
     * Only owned cells are reordered; halo cells remain at the end.
     *
     * @param strategy The reordering strategy (e.g., "rcm", "gps", "sloan")
     * @param restore If true, restores the original ordering instead
     */
    void reorderCells(const std::string& strategy = "rcm", bool restore = false);

    /**
     * @brief Reorders the nodes of the mesh to improve locality.
     *
     * @param strategy The reordering strategy (e.g., "rcm", "spatial_x")
     * @param restore If true, restores the original ordering instead
     */
    void reorderNodes(const std::string& strategy = "rcm", bool restore = false);

private:
    // =========================================================================
    // Internal State for Order Restoration
    // =========================================================================

    bool hasStoredOrder_ = false;

    std::vector<std::size_t> originalL2gCells_;
    std::unordered_map<std::size_t, std::size_t> originalG2lCells_;
    std::vector<std::size_t> originalL2gNodes_;
    std::unordered_map<std::size_t, std::size_t> originalG2lNodes_;
    std::vector<std::array<double, 3>> originalNodeCoords_;
    std::vector<std::vector<std::size_t>> originalCellNodeConnectivity_;
    std::vector<int> originalCellElementTypes_;
    std::map<int, std::vector<std::size_t>> originalSendMap_;
    std::map<int, std::vector<std::size_t>> originalRecvMap_;

    // =========================================================================
    // Internal Methods
    // =========================================================================

    /**
     * @brief Populates the LocalMesh with data from the global mesh.
     */
    void buildFromGlobalMesh(const PolyMesh& globalMesh);

    /**
     * @brief Populates local boundary face information from global mesh.
     */
    void populateBoundaryFaces(const PolyMesh& globalMesh);

    /**
     * @brief Stores the current ordering for later restoration.
     */
    void storeOriginalOrdering();

    /**
     * @brief Restores the original cell ordering.
     */
    void restoreOriginalCellOrdering();

    /**
     * @brief Restores the original node ordering.
     */
    void restoreOriginalNodeOrdering();
};

}  // namespace fvm
