# FVM Mesh

A Python package for mesh generation and analysis for Finite Volume Methods (FVM).

This package provides tools for creating, partitioning, and analyzing unstructured meshes for use in FVM simulations. It is built on top of the powerful Gmsh mesh generator and provides a simple and intuitive interface for generating high-quality meshes.

## Features

- **Mesh Generation:** Generate 2D triangular, quadrilateral, and mixed-element meshes using Gmsh.
- **Mesh Partitioning:** Partition meshes for parallel computing using METIS.
- **Mesh Analysis:** Analyze mesh quality and compute mesh statistics.
- **Local Mesh Creation:** Create local meshes for each partition for use in parallel solvers.
- **Mesh Reordering:** Reorder cells and nodes to improve cache efficiency.

## Installation

To install the package, you can use `pip`:

```bash
pip install fvm_mesh
```

## Development Installation

To install the package for development, you can clone the repository and install it in editable mode. This allows you to make changes to the source code and have them reflected immediately without reinstalling.

First, ensure you have the necessary build tools:

```bash
pip install build
```

Next, build the package from the project's root directory:

```bash
python -m build
```

Finally, install the package in editable mode:

```bash
pip install -e .
```

## Usage

Here is a simple example of how to use the package to generate a triangular mesh:

```python
from fvm_mesh.meshgen.geometry import Geometry
from fvm_mesh.meshgen.mesh_generator import MeshGenerator

# Create a rectangular geometry
geom = Geometry()
surface_tag = geom.rectangle(length=1, width=1, mesh_size=0.1)

# Generate a triangular mesh
mesher = MeshGenerator(surface_tags=surface_tag)
mesh_params = {surface_tag: {"mesh_type": "tri", "char_length": 0.1}}
mesher.generate(mesh_params=mesh_params, filename="triangular_mesh.msh")
```

## Testing

To run the tests, you can use the `unittest` module:

```bash
python -m unittest discover tests
```
