#include "reorder.hpp"
#include "poly_mesh.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <stdexcept>
#include <unordered_set>

namespace fvm {

// =========================================================================
// Utility Functions
// =========================================================================

SparseMatrix buildCellAdjacency(const PolyMesh& mesh, std::size_t numCells) {
    if (numCells == 0) numCells = mesh.nCells;
    if (numCells == 0) return SparseMatrix(0);

    // First pass: count neighbors for each cell
    std::vector<std::size_t> neighborCounts(numCells, 0);
    for (std::size_t i = 0; i < numCells; ++i) {
        for (int neighbor : mesh.cellNeighbors[i]) {
            if (neighbor >= 0 && static_cast<std::size_t>(neighbor) < numCells &&
                static_cast<std::size_t>(neighbor) != i) {
                neighborCounts[i]++;
            }
        }
    }

    // Build CSR structure
    SparseMatrix adj(numCells);
    adj.rowPtr[0] = 0;
    for (std::size_t i = 0; i < numCells; ++i) {
        adj.rowPtr[i + 1] = adj.rowPtr[i] + neighborCounts[i];
    }

    adj.colIdx.resize(adj.rowPtr[numCells]);
    std::vector<std::size_t> currentPos = adj.rowPtr;

    for (std::size_t i = 0; i < numCells; ++i) {
        for (int neighbor : mesh.cellNeighbors[i]) {
            if (neighbor >= 0 && static_cast<std::size_t>(neighbor) < numCells &&
                static_cast<std::size_t>(neighbor) != i) {
                adj.colIdx[currentPos[i]++] = static_cast<std::size_t>(neighbor);
            }
        }
    }

    return adj;
}

SparseMatrix buildNodeAdjacency(const PolyMesh& mesh) {
    if (mesh.nNodes == 0) return SparseMatrix(0);

    // Build adjacency using sets first
    std::vector<std::set<std::size_t>> adjSets(mesh.nNodes);

    for (const auto& cell : mesh.cellNodeConnectivity) {
        for (std::size_t i = 0; i < cell.size(); ++i) {
            for (std::size_t j = i + 1; j < cell.size(); ++j) {
                adjSets[cell[i]].insert(cell[j]);
                adjSets[cell[j]].insert(cell[i]);
            }
        }
    }

    // Convert to CSR
    SparseMatrix adj(mesh.nNodes);
    adj.rowPtr[0] = 0;
    for (std::size_t i = 0; i < mesh.nNodes; ++i) {
        adj.rowPtr[i + 1] = adj.rowPtr[i] + adjSets[i].size();
    }

    adj.colIdx.reserve(adj.rowPtr[mesh.nNodes]);
    for (std::size_t i = 0; i < mesh.nNodes; ++i) {
        for (std::size_t j : adjSets[i]) {
            adj.colIdx.push_back(j);
        }
    }

    return adj;
}

std::unique_ptr<CellReorderStrategy> createCellReorderStrategy(const std::string& name) {
    if (name == "rcm") return std::make_unique<RCMStrategy>();
    if (name == "gps") return std::make_unique<GPSStrategy>();
    if (name == "sloan") return std::make_unique<SloanStrategy>();
    if (name == "spectral") return std::make_unique<SpectralStrategy>();
    if (name == "spatial_x") return std::make_unique<SpatialXStrategy>();
    if (name == "spatial_y") return std::make_unique<SpatialYStrategy>();
    if (name == "random") return std::make_unique<RandomStrategy>();

    throw std::invalid_argument("Unknown cell reorder strategy: " + name);
}

// =========================================================================
// RCM Strategy
// =========================================================================

std::vector<std::size_t> RCMStrategy::getOrder(const PolyMesh& mesh) {
    SparseMatrix adj = buildCellAdjacency(mesh);
    std::size_t n = adj.nRows;

    if (n == 0) return {};

    // Find starting node with minimum degree
    std::size_t startNode = 0;
    std::size_t minDegree = std::numeric_limits<std::size_t>::max();
    for (std::size_t i = 0; i < n; ++i) {
        std::size_t deg = adj.degree(i);
        if (deg < minDegree) {
            minDegree = deg;
            startNode = i;
        }
    }

    // BFS traversal with neighbor sorting by degree
    std::vector<std::size_t> order;
    order.reserve(n);
    std::vector<bool> visited(n, false);

    std::queue<std::size_t> queue;
    queue.push(startNode);
    visited[startNode] = true;

    while (order.size() < n) {
        // Process current queue
        while (!queue.empty()) {
            std::size_t current = queue.front();
            queue.pop();
            order.push_back(current);

            // Get neighbors sorted by degree
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

        // Handle disconnected components
        if (order.size() < n) {
            for (std::size_t i = 0; i < n; ++i) {
                if (!visited[i]) {
                    queue.push(i);
                    visited[i] = true;
                    break;
                }
            }
        }
    }

    // Reverse for RCM
    std::reverse(order.begin(), order.end());
    return order;
}

// =========================================================================
// GPS Strategy
// =========================================================================

std::vector<std::size_t> GPSStrategy::getOrder(const PolyMesh& mesh) {
    SparseMatrix adj = buildCellAdjacency(mesh);
    std::size_t n = adj.nRows;

    if (n == 0) return {};

    // Helper: BFS to find distances and farthest node
    auto bfs = [&adj, n](std::size_t start) {
        std::vector<int> dist(n, -1);
        std::queue<std::size_t> q;
        q.push(start);
        dist[start] = 0;

        std::size_t farthest = start;
        int maxDist = 0;

        while (!q.empty()) {
            std::size_t curr = q.front();
            q.pop();

            auto [begin, end] = adj.neighbors(curr);
            for (auto it = begin; it != end; ++it) {
                if (dist[*it] == -1) {
                    dist[*it] = dist[curr] + 1;
                    q.push(*it);
                    if (dist[*it] > maxDist) {
                        maxDist = dist[*it];
                        farthest = *it;
                    }
                }
            }
        }

        return std::make_pair(farthest, dist);
    };

    // Find starting node with minimum degree
    std::size_t startNode = 0;
    std::size_t minDegree = std::numeric_limits<std::size_t>::max();
    for (std::size_t i = 0; i < n; ++i) {
        std::size_t deg = adj.degree(i);
        if (deg < minDegree) {
            minDegree = deg;
            startNode = i;
        }
    }

    // Find pseudo-peripheral nodes
    auto [x, dist1] = bfs(startNode);
    auto [y, dist2] = bfs(x);

    // Find center of path from x to y using BFS predecessors
    std::vector<int> predecessor(n, -1);
    std::queue<std::size_t> q;
    q.push(x);
    predecessor[x] = static_cast<int>(x);

    while (!q.empty()) {
        std::size_t curr = q.front();
        q.pop();

        if (curr == y) break;

        auto [begin, end] = adj.neighbors(curr);
        for (auto it = begin; it != end; ++it) {
            if (predecessor[*it] == -1) {
                predecessor[*it] = static_cast<int>(curr);
                q.push(*it);
            }
        }
    }

    // Trace path from y to x
    std::vector<std::size_t> path;
    std::size_t curr = y;
    while (curr != x) {
        path.push_back(curr);
        if (predecessor[curr] < 0) break;
        curr = static_cast<std::size_t>(predecessor[curr]);
    }
    path.push_back(x);

    // Center node
    std::size_t centerNode = path.empty() ? x : path[path.size() / 2];

    // BFS from center with degree-sorted neighbors
    std::vector<std::size_t> order;
    order.reserve(n);
    std::vector<bool> visited(n, false);

    std::queue<std::size_t> queue;
    queue.push(centerNode);
    visited[centerNode] = true;

    while (order.size() < n) {
        while (!queue.empty()) {
            std::size_t current = queue.front();
            queue.pop();
            order.push_back(current);

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

        // Handle disconnected components
        if (order.size() < n) {
            for (std::size_t i = 0; i < n; ++i) {
                if (!visited[i]) {
                    queue.push(i);
                    visited[i] = true;
                    break;
                }
            }
        }
    }

    // Reverse for RCM-like ordering
    std::reverse(order.begin(), order.end());
    return order;
}

// =========================================================================
// Sloan Strategy
// =========================================================================

std::vector<std::size_t> SloanStrategy::getOrder(const PolyMesh& mesh) {
    SparseMatrix adj = buildCellAdjacency(mesh);
    std::size_t n = adj.nRows;

    if (n == 0) return {};

    // Find endpoints using BFS
    auto findFarthest = [&adj, n](std::size_t start) {
        std::vector<int> dist(n, -1);
        std::queue<std::size_t> q;
        q.push(start);
        dist[start] = 0;

        std::size_t farthest = start;
        int maxDist = 0;

        while (!q.empty()) {
            std::size_t curr = q.front();
            q.pop();

            auto [begin, end] = adj.neighbors(curr);
            for (auto it = begin; it != end; ++it) {
                if (dist[*it] == -1) {
                    dist[*it] = dist[curr] + 1;
                    q.push(*it);
                    if (dist[*it] > maxDist) {
                        maxDist = dist[*it];
                        farthest = *it;
                    }
                }
            }
        }

        return std::make_pair(farthest, dist);
    };

    // Find starting and ending nodes
    std::size_t startNode = 0;
    std::size_t minDegree = std::numeric_limits<std::size_t>::max();
    for (std::size_t i = 0; i < n; ++i) {
        if (adj.degree(i) < minDegree) {
            minDegree = adj.degree(i);
            startNode = i;
        }
    }

    auto [endNode, distances] = findFarthest(startNode);

    // Compute distances from end node
    std::vector<int> distFromEnd(n, 0);
    {
        std::queue<std::size_t> q;
        q.push(endNode);
        std::vector<bool> visited(n, false);
        visited[endNode] = true;

        while (!q.empty()) {
            std::size_t curr = q.front();
            q.pop();

            auto [begin, end] = adj.neighbors(curr);
            for (auto it = begin; it != end; ++it) {
                if (!visited[*it]) {
                    visited[*it] = true;
                    distFromEnd[*it] = distFromEnd[curr] + 1;
                    q.push(*it);
                }
            }
        }
    }

    // Sloan algorithm
    std::vector<std::size_t> perm;
    perm.reserve(n);

    std::vector<int> status(n, 0);  // 0: inactive, 1: preactive, 2: numbered
    std::vector<int> currentDegrees(n);
    for (std::size_t i = 0; i < n; ++i) {
        currentDegrees[i] = static_cast<int>(adj.degree(i));
    }

    status[startNode] = 1;  // Start as preactive

    while (perm.size() < n) {
        // Find preactive nodes
        std::vector<std::size_t> preactiveNodes;
        for (std::size_t i = 0; i < n; ++i) {
            if (status[i] == 1) preactiveNodes.push_back(i);
        }

        if (preactiveNodes.empty()) {
            // Find new starting node from inactive nodes
            for (std::size_t i = 0; i < n; ++i) {
                if (status[i] == 0) {
                    status[i] = 1;
                    preactiveNodes.push_back(i);
                    break;
                }
            }
            if (preactiveNodes.empty()) break;
        }

        // Find node with highest priority (distance - 2*degree)
        std::size_t bestNode = preactiveNodes[0];
        int bestPriority = distFromEnd[bestNode] - 2 * currentDegrees[bestNode];

        for (std::size_t node : preactiveNodes) {
            int priority = distFromEnd[node] - 2 * currentDegrees[node];
            if (priority > bestPriority) {
                bestPriority = priority;
                bestNode = node;
            }
        }

        // Number this node
        perm.push_back(bestNode);
        status[bestNode] = 2;

        // Update neighbors
        auto [begin, end] = adj.neighbors(bestNode);
        for (auto it = begin; it != end; ++it) {
            if (status[*it] == 0) {
                status[*it] = 1;  // Make preactive
            }
            if (status[*it] != 2) {
                currentDegrees[*it]--;
            }
        }
    }

    return perm;
}

// =========================================================================
// Spectral Strategy
// =========================================================================

std::vector<std::size_t> SpectralStrategy::getOrder(const PolyMesh& mesh) {
    SparseMatrix adj = buildCellAdjacency(mesh);
    std::size_t n = adj.nRows;

    if (n == 0) return {};
    if (n == 1) return {0};

    // Compute Fiedler vector using power iteration on (D - A)
    // We use inverse iteration to find the second smallest eigenvector

    // Build degree vector
    std::vector<double> degree(n);
    for (std::size_t i = 0; i < n; ++i) {
        degree[i] = static_cast<double>(adj.degree(i));
    }

    // Initialize random vector orthogonal to constant vector
    std::vector<double> x(n);
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = dis(gen);
    }

    // Remove component along constant vector
    double mean = std::accumulate(x.begin(), x.end(), 0.0) / n;
    for (std::size_t i = 0; i < n; ++i) {
        x[i] -= mean;
    }

    // Normalize
    double norm = 0.0;
    for (double v : x) norm += v * v;
    norm = std::sqrt(norm);
    if (norm > 1e-10) {
        for (double& v : x) v /= norm;
    }

    // Power iteration to find Fiedler vector
    // We use shifted inverse iteration: solve (L + sigma*I)y = x
    // For simplicity, we use direct iteration on L with deflation

    const int maxIter = 100;
    const double tol = 1e-6;

    for (int iter = 0; iter < maxIter; ++iter) {
        // y = L * x = D*x - A*x
        std::vector<double> y(n, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            y[i] = degree[i] * x[i];
            auto [begin, end] = adj.neighbors(i);
            for (auto it = begin; it != end; ++it) {
                y[i] -= x[*it];
            }
        }

        // Remove component along constant vector (deflation)
        mean = std::accumulate(y.begin(), y.end(), 0.0) / n;
        for (std::size_t i = 0; i < n; ++i) {
            y[i] -= mean;
        }

        // Normalize
        norm = 0.0;
        for (double v : y) norm += v * v;
        norm = std::sqrt(norm);

        if (norm < 1e-10) break;

        // Check convergence
        double diff = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            double newVal = y[i] / norm;
            diff += (newVal - x[i]) * (newVal - x[i]);
            x[i] = newVal;
        }

        if (std::sqrt(diff) < tol) break;
    }

    // Sort by Fiedler vector values
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&x](std::size_t a, std::size_t b) { return x[a] < x[b]; });

    return order;
}

// =========================================================================
// Spatial Strategies
// =========================================================================

std::vector<std::size_t> SpatialXStrategy::getOrder(const PolyMesh& mesh) {
    std::size_t n = mesh.nCells;
    if (n == 0) return {};

    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), 0);

    std::sort(order.begin(), order.end(),
              [&mesh](std::size_t a, std::size_t b) {
                  return mesh.cellCentroids[a][0] < mesh.cellCentroids[b][0];
              });

    return order;
}

std::vector<std::size_t> SpatialYStrategy::getOrder(const PolyMesh& mesh) {
    std::size_t n = mesh.nCells;
    if (n == 0) return {};

    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), 0);

    std::sort(order.begin(), order.end(),
              [&mesh](std::size_t a, std::size_t b) {
                  return mesh.cellCentroids[a][1] < mesh.cellCentroids[b][1];
              });

    return order;
}

// =========================================================================
// Random Strategy
// =========================================================================

std::vector<std::size_t> RandomStrategy::getOrder(const PolyMesh& mesh) {
    std::size_t n = mesh.nCells;
    if (n < 2) {
        std::vector<std::size_t> order(n);
        std::iota(order.begin(), order.end(), 0);
        return order;
    }

    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), 0);

    std::random_device rd;
    std::mt19937 gen(rd());

    // Ensure the order is different from the original
    std::vector<std::size_t> original = order;
    do {
        std::shuffle(order.begin(), order.end(), gen);
    } while (order == original);

    return order;
}

// =========================================================================
// Public API
// =========================================================================

void renumberCells(PolyMesh& mesh, const std::string& strategy,
                   std::size_t numToReorder) {
    if (mesh.nCells == 0) return;

    if (numToReorder == 0) numToReorder = mesh.nCells;

    auto strat = createCellReorderStrategy(strategy);

    // Create a temporary mesh view if we're only reordering a subset
    std::vector<std::size_t> newOrder;
    if (numToReorder < mesh.nCells) {
        // Create temporary mesh with only owned cells for reordering
        PolyMesh tempMesh;
        tempMesh.dimension = mesh.dimension;
        tempMesh.nNodes = mesh.nNodes;
        tempMesh.nCells = numToReorder;
        tempMesh.nodeCoords = mesh.nodeCoords;

        tempMesh.cellNodeConnectivity.reserve(numToReorder);
        for (std::size_t i = 0; i < numToReorder; ++i) {
            tempMesh.cellNodeConnectivity.push_back(mesh.cellNodeConnectivity[i]);
        }

        // Need to analyze for neighbor information
        tempMesh.analyzeMesh();

        // Copy centroids if available
        if (!mesh.cellCentroids.empty()) {
            tempMesh.cellCentroids.resize(numToReorder);
            for (std::size_t i = 0; i < numToReorder; ++i) {
                tempMesh.cellCentroids[i] = mesh.cellCentroids[i];
            }
        }

        auto localOrder = strat->getOrder(tempMesh);

        // Combine with original indices for remaining cells
        newOrder = localOrder;
        for (std::size_t i = numToReorder; i < mesh.nCells; ++i) {
            newOrder.push_back(i);
        }
    } else {
        newOrder = strat->getOrder(mesh);
    }

    // Apply the reordering
    std::vector<std::vector<std::size_t>> newConnectivity(mesh.nCells);
    std::vector<int> newElementTypes(mesh.nCells);

    for (std::size_t i = 0; i < mesh.nCells; ++i) {
        newConnectivity[i] = mesh.cellNodeConnectivity[newOrder[i]];
        if (!mesh.cellElementTypes.empty()) {
            newElementTypes[i] = mesh.cellElementTypes[newOrder[i]];
        }
    }

    mesh.cellNodeConnectivity = std::move(newConnectivity);
    if (!mesh.cellElementTypes.empty()) {
        mesh.cellElementTypes = std::move(newElementTypes);
    }

    // Invalidate derived data
    mesh.cellNeighbors.clear();
    mesh.cellCentroids.clear();
    mesh.cellVolumes.clear();
}

void renumberNodes(PolyMesh& mesh, const std::string& strategy) {
    if (mesh.nNodes == 0) return;

    std::vector<std::size_t> newOrder;

    if (strategy == "rcm") {
        SparseMatrix adj = buildNodeAdjacency(mesh);
        std::size_t n = adj.nRows;

        if (n == 0) return;

        // RCM for nodes
        std::size_t startNode = 0;
        std::size_t minDegree = std::numeric_limits<std::size_t>::max();
        for (std::size_t i = 0; i < n; ++i) {
            if (adj.degree(i) < minDegree) {
                minDegree = adj.degree(i);
                startNode = i;
            }
        }

        newOrder.reserve(n);
        std::vector<bool> visited(n, false);
        std::queue<std::size_t> queue;
        queue.push(startNode);
        visited[startNode] = true;

        while (newOrder.size() < n) {
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

            if (newOrder.size() < n) {
                for (std::size_t i = 0; i < n; ++i) {
                    if (!visited[i]) {
                        queue.push(i);
                        visited[i] = true;
                        break;
                    }
                }
            }
        }

        std::reverse(newOrder.begin(), newOrder.end());
    } else if (strategy == "sequential") {
        newOrder.resize(mesh.nNodes);
        std::iota(newOrder.begin(), newOrder.end(), 0);
    } else if (strategy == "reverse") {
        newOrder.resize(mesh.nNodes);
        std::iota(newOrder.rbegin(), newOrder.rend(), 0);
    } else if (strategy == "spatial_x") {
        newOrder.resize(mesh.nNodes);
        std::iota(newOrder.begin(), newOrder.end(), 0);
        std::sort(newOrder.begin(), newOrder.end(),
                  [&mesh](std::size_t a, std::size_t b) {
                      return mesh.nodeCoords[a][0] < mesh.nodeCoords[b][0];
                  });
    } else if (strategy == "spatial_y") {
        newOrder.resize(mesh.nNodes);
        std::iota(newOrder.begin(), newOrder.end(), 0);
        std::sort(newOrder.begin(), newOrder.end(),
                  [&mesh](std::size_t a, std::size_t b) {
                      return mesh.nodeCoords[a][1] < mesh.nodeCoords[b][1];
                  });
    } else if (strategy == "random") {
        newOrder.resize(mesh.nNodes);
        std::iota(newOrder.begin(), newOrder.end(), 0);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(newOrder.begin(), newOrder.end(), gen);
    } else {
        throw std::invalid_argument("Unknown node reorder strategy: " + strategy);
    }

    // Create inverse mapping: remap[oldIdx] = newIdx
    std::vector<std::size_t> remap(mesh.nNodes);
    for (std::size_t newIdx = 0; newIdx < mesh.nNodes; ++newIdx) {
        remap[newOrder[newIdx]] = newIdx;
    }

    // Apply reordering to node coordinates
    std::vector<std::array<double, 3>> newCoords(mesh.nNodes);
    for (std::size_t i = 0; i < mesh.nNodes; ++i) {
        newCoords[i] = mesh.nodeCoords[newOrder[i]];
    }
    mesh.nodeCoords = std::move(newCoords);

    // Update cell connectivity to use new node indices
    for (auto& conn : mesh.cellNodeConnectivity) {
        for (std::size_t& nodeIdx : conn) {
            nodeIdx = remap[nodeIdx];
        }
    }

    // Invalidate derived data
    mesh.cellNeighbors.clear();
    mesh.cellCentroids.clear();
    mesh.cellFaceNodes.clear();
}

}  // namespace fvm
