#include "partition.hpp"
#include "poly_mesh.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>

// Conditionally include METIS
#ifdef USE_METIS
#include <metis.h>
#endif

namespace fvm {

// =========================================================================
// Utility Functions
// =========================================================================

std::vector<std::vector<int>> getAdjacencyList(const PolyMesh& mesh) {
    std::vector<std::vector<int>> adjacency(mesh.nCells);

    for (std::size_t i = 0; i < mesh.nCells; ++i) {
        for (int neighbor : mesh.cellNeighbors[i]) {
            if (neighbor >= 0 && neighbor != static_cast<int>(i)) {
                adjacency[i].push_back(neighbor);
            }
        }
        // Remove duplicates and sort
        std::sort(adjacency[i].begin(), adjacency[i].end());
        adjacency[i].erase(
            std::unique(adjacency[i].begin(), adjacency[i].end()),
            adjacency[i].end()
        );
    }

    return adjacency;
}

bool isMetisAvailable() {
#ifdef USE_METIS
    return true;
#else
    return false;
#endif
}

// =========================================================================
// Main Partitioning Function
// =========================================================================

std::vector<int> partitionMesh(
    const PolyMesh& mesh,
    int nParts,
    const std::string& method,
    const std::vector<double>& cellWeights
) {
    if (nParts <= 1) {
        return std::vector<int>(mesh.nCells, 0);
    }

    if (mesh.nCells == 0) {
        return {};
    }

    if (method == "metis") {
        return partitionWithMetis(mesh, nParts, cellWeights);
    } else if (method == "hierarchical") {
        return partitionWithHierarchical(mesh, nParts, cellWeights);
    } else {
        throw std::invalid_argument("Unknown partition method: " + method);
    }
}

// =========================================================================
// METIS Partitioning
// =========================================================================

std::vector<int> partitionWithMetis(
    const PolyMesh& mesh,
    int nParts,
    const std::vector<double>& cellWeights
) {
#ifdef USE_METIS
    if (mesh.nCells == 0) {
        return {};
    }

    // Build adjacency list
    auto adjacency = getAdjacencyList(mesh);

    // Convert to METIS CSR format
    idx_t nvtxs = static_cast<idx_t>(mesh.nCells);
    std::vector<idx_t> xadj(nvtxs + 1);
    std::vector<idx_t> adjncy;

    xadj[0] = 0;
    for (idx_t i = 0; i < nvtxs; ++i) {
        xadj[i + 1] = xadj[i] + static_cast<idx_t>(adjacency[i].size());
        for (int neighbor : adjacency[i]) {
            adjncy.push_back(static_cast<idx_t>(neighbor));
        }
    }

    // Prepare vertex weights if provided
    std::vector<idx_t> vwgt;
    if (!cellWeights.empty()) {
        vwgt.reserve(nvtxs);
        for (std::size_t i = 0; i < mesh.nCells; ++i) {
            // METIS requires integer weights, scale appropriately
            vwgt.push_back(static_cast<idx_t>(cellWeights[i] * 1000 + 1));
        }
    }

    // METIS parameters
    idx_t ncon = 1;
    idx_t nparts = static_cast<idx_t>(nParts);
    idx_t objval;
    std::vector<idx_t> part(nvtxs);

    // Options
    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_NUMBERING] = 0;  // C-style numbering

    // Call METIS
    int status = METIS_PartGraphKway(
        &nvtxs,
        &ncon,
        xadj.data(),
        adjncy.data(),
        vwgt.empty() ? nullptr : vwgt.data(),  // vertex weights
        nullptr,  // vertex sizes
        nullptr,  // edge weights
        &nparts,
        nullptr,  // target partition weights
        nullptr,  // ubvec
        options,
        &objval,
        part.data()
    );

    if (status != METIS_OK) {
        throw std::runtime_error("METIS partitioning failed with status: " +
                                 std::to_string(status));
    }

    return std::vector<int>(part.begin(), part.end());

#else
    // METIS not available, fall back to hierarchical
    std::cerr << "Warning: METIS not available, falling back to hierarchical partitioning\n";
    return partitionWithHierarchical(mesh, nParts, cellWeights);
#endif
}

// =========================================================================
// Hierarchical Coordinate Bisection
// =========================================================================

std::vector<int> partitionWithHierarchical(
    const PolyMesh& mesh,
    int nParts,
    const std::vector<double>& cellWeights
) {
    if (mesh.nCells == 0) {
        return {};
    }

    // Check if nParts is power of two
    bool isPowerOfTwo = (nParts > 0) && ((nParts & (nParts - 1)) == 0);
    if (!isPowerOfTwo) {
        std::cerr << "Warning: Hierarchical method works best with power-of-two "
                  << "partitions. n_parts=" << nParts << " may result in uneven partitions.\n";
    }

    // Initialize weights (uniform if not provided)
    std::vector<double> weights(mesh.nCells, 1.0);
    if (!cellWeights.empty() && cellWeights.size() == mesh.nCells) {
        weights = cellWeights;
    }

    // Initialize all cells to partition 0
    std::vector<int> parts(mesh.nCells, 0);

    // Iteratively bisect partitions until we have nParts
    for (int i = 1; i < nParts; ++i) {
        // Find the partition with the most cells to split
        std::vector<std::size_t> partCounts(i + 1, 0);
        for (int p : parts) {
            if (p < static_cast<int>(partCounts.size())) {
                partCounts[p]++;
            }
        }

        int partToSplit = static_cast<int>(
            std::max_element(partCounts.begin(), partCounts.end()) - partCounts.begin()
        );

        // Get indices of cells in this partition
        std::vector<std::size_t> idxsToSplit;
        for (std::size_t j = 0; j < mesh.nCells; ++j) {
            if (parts[j] == partToSplit) {
                idxsToSplit.push_back(j);
            }
        }

        if (idxsToSplit.empty()) continue;

        // Determine axis to split (longest dimension of bounding box)
        double minX = std::numeric_limits<double>::max();
        double maxX = std::numeric_limits<double>::lowest();
        double minY = std::numeric_limits<double>::max();
        double maxY = std::numeric_limits<double>::lowest();
        double minZ = std::numeric_limits<double>::max();
        double maxZ = std::numeric_limits<double>::lowest();

        for (std::size_t idx : idxsToSplit) {
            const auto& c = mesh.cellCentroids[idx];
            minX = std::min(minX, c[0]);
            maxX = std::max(maxX, c[0]);
            minY = std::min(minY, c[1]);
            maxY = std::max(maxY, c[1]);
            minZ = std::min(minZ, c[2]);
            maxZ = std::max(maxZ, c[2]);
        }

        double rangeX = maxX - minX;
        double rangeY = maxY - minY;
        double rangeZ = maxZ - minZ;

        int axis = 0;  // Default to X
        if (rangeY > rangeX && rangeY >= rangeZ) {
            axis = 1;
        } else if (rangeZ > rangeX && rangeZ > rangeY) {
            axis = 2;
        }

        // Sort indices by coordinate along chosen axis
        std::sort(idxsToSplit.begin(), idxsToSplit.end(),
                  [&mesh, axis](std::size_t a, std::size_t b) {
                      return mesh.cellCentroids[a][axis] < mesh.cellCentroids[b][axis];
                  });

        // Find weighted median split point
        std::vector<double> sortedWeights;
        for (std::size_t idx : idxsToSplit) {
            sortedWeights.push_back(weights[idx]);
        }

        double totalWeight = std::accumulate(sortedWeights.begin(),
                                             sortedWeights.end(), 0.0);
        double halfWeight = totalWeight / 2.0;

        std::size_t splitIdx = idxsToSplit.size() / 2;
        if (totalWeight > 0) {
            double cumWeight = 0.0;
            for (std::size_t k = 0; k < sortedWeights.size(); ++k) {
                cumWeight += sortedWeights[k];
                if (cumWeight >= halfWeight) {
                    splitIdx = k;
                    break;
                }
            }
        }

        // Handle edge cases
        if (splitIdx == 0) splitIdx = 1;
        if (splitIdx >= idxsToSplit.size()) splitIdx = idxsToSplit.size() - 1;

        // Assign cells on the right side to new partition
        for (std::size_t k = splitIdx; k < idxsToSplit.size(); ++k) {
            parts[idxsToSplit[k]] = i;
        }
    }

    return parts;
}

// =========================================================================
// Partition Summary
// =========================================================================

void printPartitionSummary(const std::vector<int>& parts) {
    if (parts.empty()) {
        std::cout << "--- Partition Summary ---\n";
        std::cout << "No partitions found.\n";
        return;
    }

    int nParts = *std::max_element(parts.begin(), parts.end()) + 1;

    // Count cells per partition
    std::vector<std::size_t> counts(nParts, 0);
    for (int p : parts) {
        if (p >= 0 && p < nParts) {
            counts[p]++;
        }
    }

    std::cout << "--- Partition Summary ---\n";
    std::cout << "Number of partitions: " << nParts << "\n";

    std::size_t totalCells = parts.size();
    std::size_t minCells = *std::min_element(counts.begin(), counts.end());
    std::size_t maxCells = *std::max_element(counts.begin(), counts.end());
    double avgCells = static_cast<double>(totalCells) / nParts;

    for (int p = 0; p < nParts; ++p) {
        double pct = 100.0 * counts[p] / totalCells;
        std::cout << "  Partition " << std::setw(3) << p << ": "
                  << std::setw(8) << counts[p] << " cells ("
                  << std::fixed << std::setprecision(1) << pct << "%)\n";
    }

    std::cout << "\nBalance Statistics:\n";
    std::cout << "  Min cells per partition: " << minCells << "\n";
    std::cout << "  Max cells per partition: " << maxCells << "\n";
    std::cout << "  Avg cells per partition: " << std::fixed
              << std::setprecision(1) << avgCells << "\n";

    double imbalance = (maxCells > 0)
                       ? (static_cast<double>(maxCells) / avgCells - 1.0) * 100.0
                       : 0.0;
    std::cout << "  Load imbalance: " << std::fixed << std::setprecision(1)
              << imbalance << "%\n";
}

}  // namespace fvm
