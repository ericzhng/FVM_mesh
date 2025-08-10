from pathlib import Path
import os
import numpy as np
from mesh import Mesh, PartitionManager, build_halo_indices_from_decomposed, mesh_print_summary, partition_print_summary

def make_simple_tet_mesh():
    m = Mesh()
    m.dimension = 3
    m.node_coords = np.array([
        [0.0,0.0,0.0],
        [1.0,0.0,0.0],
        [0.0,1.0,0.0],
        [0.0,0.0,1.0],
        [1.0,1.0,1.0],
    ])
    m.num_nodes = m.node_coords.shape[0]
    m.cell_connectivity = [[0,1,2,3],[1,2,3,4]]
    m.num_cells = 2
    m.boundary_faces_nodes = np.empty((0,3), dtype=int)
    m.boundary_faces_tags = np.empty((0,), dtype=int)
    m.analyze_mesh()
    return m

def test_empty_partitions(tmp_path):
    m = make_simple_tet_mesh()
    pm = PartitionManager(m)
    parts = pm.partition_elements(4, method='hierarchical')
    assert parts.shape[0] == m.num_cells
    out = tmp_path / 'decomp'
    pm.write_decompose_par_json_npy(str(out) if False else str(out), 4)  # call via method if desired
    # using helper to write (call the method)
    pm.write_decompose_par_json_npy(str(out), 4)
    for p in range(4):
        proc = out / f'processor{p}'
        assert proc.exists()
        assert (proc / 'mesh.json').exists()

def test_reconstruction_roundtrip(tmp_path):
    m = make_simple_tet_mesh()
    pm = PartitionManager(m)
    pm.partition_elements(2, method='hierarchical')
    out = tmp_path / 'decomp2'
    pm.write_decompose_par_json_npy(str(out), 2)
    newmesh = pm.reconstruct_par(str(out))
    assert newmesh.num_cells == m.num_cells

def test_halo_builder(tmp_path):
    m = make_simple_tet_mesh()
    pm = PartitionManager(m)
    pm.partition_elements(2, method='hierarchical')
    out = tmp_path / 'decomp3'
    pm.write_decompose_par_json_npy(str(out), 2)
    halos = build_halo_indices_from_decomposed(str(out))
    assert set(halos.keys()) <= {0,1}
    for r, info in halos.items():
        assert isinstance(info['neighbors'], dict)

def test_gmsh_writer(tmp_path):
    m = make_simple_tet_mesh()
    pm = PartitionManager(m)
    pm.partition_elements(2, method='hierarchical')
    out = tmp_path / 'decomp4'
    pm.write_decompose_par_json_npy(str(out), 2)
    pm.write_gmsh_per_processor(str(out), 2)
    for p in range(2):
        assert (out / f'processor{p}' / f'processor{p}.msh').exists()

if __name__ == '__main__':
    import tempfile
    td = tempfile.TemporaryDirectory()
    base = Path(td.name)
    test_empty_partitions(base)
    test_reconstruction_roundtrip(base)
    test_halo_builder(base)
    test_gmsh_writer(base)
    print('All tests passed.')
