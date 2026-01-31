# Mesh Processing Library - Partition, reordering, and mesh utilities

set(MESH_PROCESSING_SOURCES
    partition.cpp
    reorder.cpp
    poly_mesh.cpp
    local_mesh.cpp
    mesh_partition_manager.cpp
    mesh_quality.cpp
    vtk_writer.cpp
)

set(MESH_PROCESSING_HEADERS
    partition.hpp
    reorder.hpp
    poly_mesh.hpp
    local_mesh.hpp
    mesh_partition_manager.hpp
    mesh_quality.hpp
    vtk_writer.hpp
)

# Create static library
add_library(mesh_processing STATIC ${MESH_PROCESSING_SOURCES} ${MESH_PROCESSING_HEADERS})

# Set library properties
target_include_directories(mesh_processing
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include/mesh_processing>
    PRIVATE
        ${Gmsh_INCLUDE_DIR}
)

# Link dependencies
# mesh_processing depends on mesh_generation for shared types (mesh_types.hpp)
target_link_libraries(mesh_processing
    PUBLIC
        FVM::mesh_generation
)

# Link METIS if available (for graph partitioning)
if(HAVE_METIS)
    target_link_libraries(mesh_processing PRIVATE METIS::METIS)
    target_compile_definitions(mesh_processing PRIVATE USE_METIS)
endif()

# Set C++ standard
target_compile_features(mesh_processing PUBLIC cxx_std_17)

# Export alias for easier usage
add_library(FVM::mesh_processing ALIAS mesh_processing)
