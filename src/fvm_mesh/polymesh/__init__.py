# -*- coding: utf-8 -*-
"""
This package provides a set of tools for creating, manipulating, and analyzing
polygonal meshes, primarily for use in Finite Volume Method (FVM) simulations.

It includes classes for representing core mesh structures, partitioned meshes
for parallel processing, and utilities for mesh quality analysis and reordering.

Key modules:
- core_mesh:  Defines the basic mesh structure (nodes, cells).
- poly_mesh:  Extends the core mesh with geometric properties (volumes, normals).
- local_mesh: Represents a partitioned mesh for a single process.
- partition:  Functions for partitioning a global mesh.
- reorder:    Tools for reordering cells and nodes to optimize matrix bandwidth.
- quality:    Functions to compute mesh quality metrics.
"""

from .poly_mesh import PolyMesh
from .local_mesh import LocalMesh
from .mesh_partition_manager import MeshPartitionManager
from .reorder import renumber_nodes, renumber_cells
from .partition import partition_mesh

__all__ = [
    "PolyMesh",
    "LocalMesh",
    "MeshPartitionManager",
    "partition_mesh",
    "renumber_nodes",
    "renumber_cells",
]
