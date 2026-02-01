#pragma once

/**
 * @file reorder.hpp
 * @brief Mesh reordering tools for optimizing matrix bandwidth.
 *
 * This module provides functions for reordering cells and nodes in a mesh
 * to optimize the bandwidth of the mesh's adjacency matrix. Reducing the
 * bandwidth can significantly improve the performance of numerical solvers.
 *
 * Supported strategies:
 * - RCM (Reverse Cuthill-McKee): Most common bandwidth reduction algorithm
 * - GPS (Gibbs-Poole-Stockmeyer): Finds pseudo-peripheral starting node
 * - Sloan: Priority-based ordering with distance weighting
 * - Spectral: Uses Fiedler vector for ordering
 * - Spatial (X/Y): Sort by coordinate values
 * - Random: Random permutation for testing
 */

#include "fvm_export.hpp"
#include <memory>
#include <string>
#include <vector>

namespace fvm {

// Forward declarations
class PolyMesh;

/**
 * @brief Sparse matrix in CSR (Compressed Sparse Row) format.
 *
 * Used internally for graph algorithms on mesh adjacency.
 */
struct SparseMatrix {
    std::size_t nRows = 0;
    std::vector<std::size_t> rowPtr;   // Row pointers (size: nRows + 1)
    std::vector<std::size_t> colIdx;   // Column indices

    SparseMatrix() = default;
    SparseMatrix(std::size_t n) : nRows(n), rowPtr(n + 1, 0) {}

    /// Returns the degree (number of neighbors) for a given row
    std::size_t degree(std::size_t row) const {
        return rowPtr[row + 1] - rowPtr[row];
    }

    /// Returns iterators to the neighbors of a given row
    auto neighbors(std::size_t row) const {
        return std::make_pair(
            colIdx.begin() + rowPtr[row],
            colIdx.begin() + rowPtr[row + 1]
        );
    }
};

/**
 * @brief Abstract base class for cell reordering strategies.
 */
class FVM_API CellReorderStrategy {
public:
    virtual ~CellReorderStrategy() = default;

    /**
     * @brief Computes the new cell order for the given mesh.
     * @param mesh The PolyMesh object
     * @return A permutation array where result[i] is the old index of new cell i
     */
    virtual std::vector<std::size_t> getOrder(const PolyMesh& mesh) = 0;
};

/**
 * @brief Reverse Cuthill-McKee (RCM) ordering strategy.
 *
 * RCM is a widely used algorithm for reducing the bandwidth of sparse
 * symmetric matrices. It starts from a node with low degree and visits
 * neighbors in degree order.
 */
class FVM_API RCMStrategy : public CellReorderStrategy {
public:
    std::vector<std::size_t> getOrder(const PolyMesh& mesh) override;
};

/**
 * @brief Gibbs-Poole-Stockmeyer (GPS) ordering strategy.
 *
 * The GPS algorithm finds a "good" starting node by locating a
 * pseudo-peripheral node, then applies a Cuthill-McKee ordering.
 */
class FVM_API GPSStrategy : public CellReorderStrategy {
public:
    std::vector<std::size_t> getOrder(const PolyMesh& mesh) override;
};

/**
 * @brief Sloan ordering strategy.
 *
 * Uses a priority queue with distance-based weighting to select nodes
 * during the ordering process.
 */
class FVM_API SloanStrategy : public CellReorderStrategy {
public:
    std::vector<std::size_t> getOrder(const PolyMesh& mesh) override;
};

/**
 * @brief Spectral bisection ordering strategy.
 *
 * Uses the Fiedler vector (second smallest eigenvector of the graph
 * Laplacian) to order nodes. Implemented using power iteration.
 */
class FVM_API SpectralStrategy : public CellReorderStrategy {
public:
    std::vector<std::size_t> getOrder(const PolyMesh& mesh) override;
};

/**
 * @brief Spatial ordering by X coordinate.
 */
class FVM_API SpatialXStrategy : public CellReorderStrategy {
public:
    std::vector<std::size_t> getOrder(const PolyMesh& mesh) override;
};

/**
 * @brief Spatial ordering by Y coordinate.
 */
class FVM_API SpatialYStrategy : public CellReorderStrategy {
public:
    std::vector<std::size_t> getOrder(const PolyMesh& mesh) override;
};

/**
 * @brief Random permutation ordering.
 */
class FVM_API RandomStrategy : public CellReorderStrategy {
public:
    std::vector<std::size_t> getOrder(const PolyMesh& mesh) override;
};

// =========================================================================
// Public API
// =========================================================================

/**
 * @brief Renumbers the cells of a mesh in-place to optimize matrix bandwidth.
 *
 * This function modifies the mesh object directly. After renumbering,
 * derived properties like neighbors and centroids are invalidated and
 * will need to be re-computed via analyzeMesh().
 *
 * @param mesh The input PolyMesh object to be modified
 * @param strategy The renumbering strategy: "rcm", "gps", "sloan",
 *                 "spectral", "spatial_x", "spatial_y", "random"
 * @param numToReorder Number of cells to reorder (0 = all cells)
 */
FVM_API void renumberCells(PolyMesh& mesh, const std::string& strategy = "rcm",
                   std::size_t numToReorder = 0);

/**
 * @brief Renumbers the nodes of a mesh in-place to optimize matrix bandwidth.
 *
 * @param mesh The input PolyMesh object to be modified
 * @param strategy The renumbering strategy: "rcm", "sequential",
 *                 "reverse", "spatial_x", "spatial_y", "random"
 */
FVM_API void renumberNodes(PolyMesh& mesh, const std::string& strategy = "rcm");

// =========================================================================
// Utility Functions
// =========================================================================

/**
 * @brief Builds the cell-to-cell adjacency matrix from mesh neighbors.
 * @param mesh The PolyMesh object
 * @param numCells Number of cells to include (0 = all cells)
 * @return Sparse adjacency matrix in CSR format
 */
FVM_API SparseMatrix buildCellAdjacency(const PolyMesh& mesh, std::size_t numCells = 0);

/**
 * @brief Builds the node-to-node adjacency matrix from cell connectivity.
 * @param mesh The PolyMesh object
 * @return Sparse adjacency matrix in CSR format
 */
FVM_API SparseMatrix buildNodeAdjacency(const PolyMesh& mesh);

/**
 * @brief Creates a reorder strategy by name.
 * @param name Strategy name
 * @return Unique pointer to the strategy
 */
FVM_API std::unique_ptr<CellReorderStrategy> createCellReorderStrategy(const std::string& name);

}  // namespace fvm
