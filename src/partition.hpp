#pragma once

/**
 * @file partition.hpp
 * @brief Mesh partitioning tools for distributed computing.
 *
 * This module provides functions for partitioning an unstructured mesh into
 * multiple subdomains. Partitioning is a crucial step for parallel processing
 * of large meshes.
 *
 * Supported methods:
 * - METIS: Graph-based partitioning using the METIS library
 * - Hierarchical: Coordinate bisection without external dependencies
 */

#include <string>
#include <vector>

namespace fvm {

// Forward declaration
class PolyMesh;

/**
 * @brief Partitions mesh elements into a specified number of parts.
 *
 * This function supports different partitioning methods, including METIS
 * and a simple hierarchical coordinate bisection method.
 *
 * @param mesh The mesh object to partition (must be analyzed)
 * @param nParts The number of partitions
 * @param method The partitioning method: "metis" or "hierarchical"
 * @param cellWeights Optional weights for each cell (empty for uniform)
 * @return A vector of partition IDs for each cell
 * @throws std::runtime_error if partitioning fails
 */
std::vector<int> partitionMesh(
    const PolyMesh& mesh,
    int nParts,
    const std::string& method = "metis",
    const std::vector<double>& cellWeights = {}
);

/**
 * @brief Partitions the mesh using the METIS library.
 *
 * Uses METIS_PartGraphKway for graph-based partitioning to minimize
 * edge cuts between partitions.
 *
 * @param mesh The mesh object to partition
 * @param nParts The number of partitions
 * @param cellWeights Optional cell weights
 * @return A vector of partition IDs for each cell
 * @throws std::runtime_error if METIS is not available or fails
 */
std::vector<int> partitionWithMetis(
    const PolyMesh& mesh,
    int nParts,
    const std::vector<double>& cellWeights = {}
);

/**
 * @brief Partitions the mesh using hierarchical coordinate bisection.
 *
 * Recursively bisects the mesh along the longest dimension until the
 * desired number of partitions is reached. Works best with power-of-two
 * partition counts.
 *
 * @param mesh The mesh object to partition
 * @param nParts The number of partitions
 * @param cellWeights Optional cell weights
 * @return A vector of partition IDs for each cell
 */
std::vector<int> partitionWithHierarchical(
    const PolyMesh& mesh,
    int nParts,
    const std::vector<double>& cellWeights = {}
);

/**
 * @brief Prints a summary of the cell distribution across partitions.
 * @param parts The partition assignment vector
 */
void printPartitionSummary(const std::vector<int>& parts);

/**
 * @brief Computes the adjacency list for mesh cells.
 *
 * Extracts the cell-to-cell connectivity from the mesh for use in
 * graph partitioning.
 *
 * @param mesh The mesh object
 * @return A vector of neighbor lists for each cell
 */
std::vector<std::vector<int>> getAdjacencyList(const PolyMesh& mesh);

/**
 * @brief Checks if METIS library is available.
 * @return true if METIS can be used
 */
bool isMetisAvailable();

}  // namespace fvm
