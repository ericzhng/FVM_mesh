# FVM Mesh Library (C++)

A comprehensive C++ mesh library for 2D/3D finite volume method (FVM) solvers. This library provides mesh generation, partitioning, quality analysis, and reordering capabilities suitable for parallel FVM solvers.

## Features

### Core Mesh Handling
- **PolyMesh**: Core data structure for unstructured polygonal meshes
- **Gmsh Integration**: Read mesh files from Gmsh `.msh` format
- **Mesh Analysis**: Automatic computation of topology and geometric properties

### Geometry Creation
- Rectangle, circle, polygon, triangle, ellipse
- Triangular, quadrilateral, structured mesh types
- Automatic boundary identification and labeling

### Mesh Partitioning
- **METIS Partitioning**: Graph-based partitioning using the METIS library
- **Hierarchical Bisection**: Coordinate-based partitioning (no external deps)
- **LocalMesh**: Partition mesh with halo cell management for distributed computing

### Mesh Reordering
- **RCM**: Reverse Cuthill-McKee for bandwidth reduction
- **GPS**: Gibbs-Poole-Stockmeyer ordering
- **Sloan**: Priority-based ordering with distance weighting
- **Spectral**: Fiedler vector-based ordering
- **Spatial**: Sort by coordinate values (X/Y)
- **Random**: Random permutation for testing

### Mesh Quality
- **Volume Ratio**: Min/max cell volume analysis
- **Skewness**: Cell shape quality metric
- **Aspect Ratio**: Edge length uniformity
- **Non-Orthogonality**: Face-to-centroid alignment
- **Connectivity Checks**: Topological validation

### Output Formats
- Gmsh MSH format (`.msh`)
- VTK Legacy format (`.vtk`)
- VTK XML format (`.vtu`) - ideal for ParaView
- OpenFOAM polyMesh format
- Boundary information file

## Requirements

- C++17 compatible compiler
- CMake 3.16+
- [Gmsh SDK](https://gmsh.info/#Download) (version 4.x)
- [METIS](http://glaros.dtc.umn.edu/gkhome/metis/metis/overview) (optional, for graph partitioning)

## Building

### Windows (Visual Studio)

```bash
# Set environment variables
set GMSH_DIR=C:\path\to\gmsh
set METIS_DIR=C:\path\to\metis  # Optional

# Configure and build
cd cpp
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=%GMSH_DIR%
cmake --build . --config Release
```

### Linux/macOS

```bash
# Set environment variables
export GMSH_DIR=/path/to/gmsh
export METIS_DIR=/path/to/metis  # Optional

# Configure and build
cd cpp
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=$GMSH_DIR
make -j$(nproc)
```

### CMake Options

| Option | Description | Default |
|--------|-------------|---------|
| `USE_METIS` | Enable METIS library support | ON |

## Usage Examples

### Basic Mesh Operations

```cpp
#include "poly_mesh.hpp"
#include "mesh_quality.hpp"

int main() {
    // Load mesh from Gmsh file
    auto mesh = fvm::PolyMesh::fromGmsh("mesh.msh");
    mesh.analyzeMesh();
    mesh.printSummary();

    // Compute quality metrics
    auto quality = fvm::MeshQuality::fromMesh(mesh);
    quality.printSummary();

    return 0;
}
```

### Mesh Partitioning

```cpp
#include "poly_mesh.hpp"
#include "mesh_partition_manager.hpp"

int main() {
    auto mesh = fvm::PolyMesh::createStructuredQuadMesh(100, 100);

    // Partition into 4 parts with cell reordering
    auto localMeshes = fvm::MeshPartitionManager::createLocalMeshes(
        mesh,
        4,              // number of partitions
        "metis",        // partitioning method
        "rcm",          // cell reordering strategy
        "rcm"           // node reordering strategy
    );

    for (const auto& lm : localMeshes) {
        std::cout << "Rank " << lm.rank << ": "
                  << lm.numOwnedCells << " owned, "
                  << lm.numHaloCells << " halo cells\n";
    }

    return 0;
}
```

### Mesh Reordering

```cpp
#include "poly_mesh.hpp"
#include "reorder.hpp"

int main() {
    auto mesh = fvm::PolyMesh::createStructuredQuadMesh(50, 50);

    // Reorder cells using RCM algorithm
    fvm::renumberCells(mesh, "rcm");

    // Re-analyze after reordering
    mesh.analyzeMesh();

    return 0;
}
```

### Direct Partitioning

```cpp
#include "poly_mesh.hpp"
#include "partition.hpp"

int main() {
    auto mesh = fvm::PolyMesh::createStructuredQuadMesh(100, 100);

    // Get partition assignments
    auto parts = fvm::partitionMesh(mesh, 8, "metis");
    fvm::printPartitionSummary(parts);

    // Or use hierarchical method (no METIS required)
    auto parts2 = fvm::partitionMesh(mesh, 8, "hierarchical");

    return 0;
}
```

### Mesh Generation

```cpp
#include "geometry.hpp"
#include "mesh_generator.hpp"
#include "vtk_writer.hpp"
#include <gmsh.h>

int main() {
    gmsh::initialize();
    gmsh::model::add("my_mesh");

    // Create geometry
    fvm::Geometry geom("My Domain");
    int surface = geom.rectangle(1.0, 1.0);

    // Set up boundary conditions
    gmsh::model::geo::synchronize();

    // Generate mesh
    fvm::MeshGenerator mesher(surface, "output");
    std::map<int, fvm::MeshParams> params;
    params[surface] = {fvm::MeshType::Quads, 0.01};
    mesher.generate(params, "mesh.msh");

    // Export to VTK
    fvm::VTKWriter::writeVTK(mesher.getMeshData(), "output/mesh.vtk");
    fvm::VTKWriter::writeVTU(mesher.getMeshData(), "output/mesh.vtu");

    gmsh::finalize();
    return 0;
}
```

## Library Structure

```
cpp/
├── include/
│   ├── poly_mesh.hpp              # Core mesh data structure
│   ├── local_mesh.hpp             # Partitioned mesh for distributed computing
│   ├── mesh_partition_manager.hpp # Partitioning orchestrator
│   ├── partition.hpp              # Partitioning algorithms
│   ├── reorder.hpp                # Cell/node reordering
│   ├── mesh_quality.hpp           # Quality metrics
│   ├── geometry.hpp               # Geometry creation utilities
│   ├── mesh_generator.hpp         # Mesh generation
│   ├── vtk_writer.hpp             # VTK export
│   └── mesh_types.hpp             # Common type definitions
├── src/
│   ├── poly_mesh.cpp
│   ├── local_mesh.cpp
│   ├── mesh_partition_manager.cpp
│   ├── partition.cpp
│   ├── reorder.cpp
│   ├── mesh_quality.cpp
│   ├── geometry.cpp
│   ├── mesh_generator.cpp
│   ├── vtk_writer.cpp
│   └── main.cpp
└── CMakeLists.txt
```

## Data Structures

### PolyMesh

The core mesh container with the following key data:

```cpp
class PolyMesh {
    int dimension;                                    // 2D or 3D
    size_t nNodes, nCells;                           // Counts

    // Topology
    std::vector<std::array<double, 3>> nodeCoords;   // Node coordinates
    std::vector<std::vector<size_t>> cellNodeConnectivity;  // Cell->nodes
    std::vector<std::vector<int>> cellNeighbors;     // Cell->neighbor cells
    std::vector<std::vector<int>> cellFaceTags;      // Face boundary tags

    // Geometry
    std::vector<std::array<double, 3>> cellCentroids;
    std::vector<double> cellVolumes;
    std::vector<std::vector<std::array<double, 3>>> cellFaceNormals;
    std::vector<std::vector<double>> cellFaceAreas;
};
```

### LocalMesh

Extends PolyMesh with partition-specific data:

```cpp
class LocalMesh : public PolyMesh {
    int rank;                                         // Partition ID
    size_t numOwnedCells, numHaloCells;              // Cell counts

    std::vector<size_t> l2gCells, l2gNodes;          // Local->global maps
    std::unordered_map<size_t, size_t> g2lCells;     // Global->local maps

    std::map<int, std::vector<size_t>> sendMap;      // Halo send targets
    std::map<int, std::vector<size_t>> recvMap;      // Halo receive sources
};
```

## Integration with Parallel Solvers

The library provides data structures suitable for:

1. **Domain Decomposition**: Use `MeshPartitionManager` to create local meshes
2. **Halo Exchange**: `sendMap` and `recvMap` define communication patterns
3. **Load Balancing**: METIS optimizes for balanced partition sizes
4. **Matrix Bandwidth**: Reordering reduces sparse matrix bandwidth

## Command Line Tool

```bash
# Basic usage
./FVM_Mesh

# With options
./FVM_Mesh --length 2.0 --height 1.0 --char-length 0.02 --output my_mesh

# Show Gmsh GUI after generation
./FVM_Mesh --gui

# Show help
./FVM_Mesh --help
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--length <value>` | Rectangle length | 1.0 |
| `--height <value>` | Rectangle height | 1.0 |
| `--mesh-size <value>` | Initial mesh size | 0.05 |
| `--char-length <value>` | Characteristic length | 0.01 |
| `--output <dir>` | Output directory | data |
| `--gui` | Show Gmsh GUI | off |

## Output Files

After running, the following files are generated:

```
data/
├── sample_rect_mesh.msh    # Gmsh native format
├── sample_rect_mesh.vtk    # VTK legacy format
├── sample_rect_mesh.vtu    # VTK XML format (ParaView)
├── boundary_info.txt       # Boundary condition info
└── openfoam/
    └── constant/
        └── polyMesh/
            ├── points      # Node coordinates
            ├── faces       # Face connectivity
            ├── owner       # Face owner cells
            ├── neighbour   # Face neighbour cells
            └── boundary    # Boundary patch info
```

## License

MIT License
