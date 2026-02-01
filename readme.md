# FVM Mesh

A Python package for mesh generation and analysis for Finite Volume Methods (FVM).

This package provides tools for creating, partitioning, and analyzing unstructured meshes for use in FVM simulations. It is built on top of the powerful Gmsh mesh generator and provides a simple and intuitive interface for generating high-quality meshes.

## Features

### Mesh Generation (`fvm_mesh.meshgen`)
- Create 2D geometries: rectangles, circles, ellipses, triangles, and arbitrary polygons
- Generate triangular, quadrilateral, and mixed-element meshes using Gmsh
- Support for geometry visualization and bounding box computation

### Mesh Analysis (`fvm_mesh.polymesh`)
- **PolyMesh**: Core data structure for unstructured polygonal meshes
  - Read meshes from Gmsh `.msh` files
  - Compute geometric properties: cell volumes, face areas, normals, centroids
  - Compute topological properties: cell neighbors, face connectivity
  - Mesh quality analysis and statistics
  - 2D mesh visualization with partition coloring

- **Mesh Partitioning**: Partition meshes for parallel computing
  - METIS-based graph partitioning
  - Hierarchical coordinate bisection method
  - Support for cell weights

- **LocalMesh**: Manage partitioned meshes for distributed computing
  - Create local meshes from global partitioned mesh
  - Handle halo cells for inter-process communication
  - Maintain global-to-local and local-to-global index mappings

- **Mesh Reordering**: Optimize memory access patterns
  - Reverse Cuthill-McKee (RCM) reordering for cells and nodes
  - Improve cache efficiency for FVM solvers

## Installation

To install the package, you can use `pip`:

```bash
pip install fvm_mesh
```

## Development Installation

To install the package for development, clone the repository and install in editable mode:

```bash
git clone <repository-url>
cd FVM_mesh
pip install -e .
```

## Building and Distributing

To build a distributable package (wheel and source distribution):

```bash
pip install build
python -m build
```

This creates `.whl` and `.tar.gz` files in the `dist/` folder.

To install from a built package:

```bash
pip install dist/fvm_mesh-<version>-py3-none-any.whl
```

To upload to PyPI (for publishing):

```bash
pip install twine
twine upload dist/*
```

## Dependencies

- `numpy` - Numerical computing
- `scipy` - Scientific computing (sparse matrices, graph algorithms)
- `gmsh` - Mesh generation
- `matplotlib` - Visualization
- `metis` - Graph partitioning (optional, for METIS partitioning method)
- `pyvista` - 3D visualization (optional)

## Usage

### Geometry Creation and Mesh Generation

```python
import gmsh
from fvm_mesh.meshgen import Geometry, MeshGenerator

gmsh.initialize()

# Create a rectangular geometry
geom = Geometry()
surface_tag = geom.rectangle(length=1.0, width=1.0, mesh_size=0.1)

# Generate a triangular mesh
mesher = MeshGenerator(surface_tags=surface_tag)
mesh_params = {surface_tag: {"mesh_type": "tri", "char_length": 0.1}}
mesher.generate(mesh_params=mesh_params, filename="mesh.msh")

gmsh.finalize()
```

### Loading and Analyzing a Mesh

```python
from fvm_mesh.polymesh import PolyMesh

# Load mesh from file
mesh = PolyMesh.from_gmsh("mesh.msh")

# Analyze the mesh (computes all geometric and topological properties)
mesh.analyze_mesh()

# Print mesh summary
mesh.print_summary()

# Plot the mesh
mesh.plot("mesh_plot.png")
```

### Mesh Partitioning

```python
from fvm_mesh.polymesh import PolyMesh, partition_mesh

# Load and analyze mesh
mesh = PolyMesh.from_gmsh("mesh.msh")
mesh.analyze_mesh()

# Partition into 4 parts using METIS
parts = partition_mesh(mesh, n_parts=4, method="metis")

# Visualize partitions
mesh.plot("partitioned_mesh.png", parts=parts)
```

### Creating Local Meshes for Parallel Computing

```python
from fvm_mesh.polymesh import PolyMesh, MeshPartitionManager

# Load and analyze mesh
mesh = PolyMesh.from_gmsh("mesh.msh")
mesh.analyze_mesh()

# Create partition manager
manager = MeshPartitionManager(mesh, n_parts=4)

# Get local mesh for a specific rank
local_mesh = manager.get_local_mesh(rank=0)

# Reorder cells for better cache performance
local_mesh.reorder_cells(strategy="rcm")
```

## Module Structure

```
fvm_mesh/
├── meshgen/
│   ├── geometry.py      # Geometry creation (Geometry class)
│   └── mesh_generator.py # Mesh generation (MeshGenerator class)
└── polymesh/
    ├── poly_mesh.py     # Core mesh data structure (PolyMesh class)
    ├── local_mesh.py    # Partitioned mesh for single process (LocalMesh class)
    ├── partition.py     # Mesh partitioning functions
    ├── reorder.py       # Cell and node reordering utilities
    ├── quality.py       # Mesh quality metrics
    └── mesh_partition_manager.py # Partition management
```

## Testing

To run the tests, use the `unittest` module:

```bash
python -m unittest discover tests
```

Or run specific test files:

```bash
python -m unittest tests.test_polymesh
python -m unittest tests.test_partition
```

## Author

**Eric Zhang**
Email: buaa.zhanghui@gmail.com

## License

This project is licensed under the MIT License - see the LICENSE file for details.
