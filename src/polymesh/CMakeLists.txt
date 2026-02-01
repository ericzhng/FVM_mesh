# Polymesh Library - Partition, reordering, and mesh utilities

set(POLYMESH_SOURCES
    partition.cpp
    reorder.cpp
    poly_mesh.cpp
    local_mesh.cpp
    mesh_partition_manager.cpp
    mesh_quality.cpp
    vtk_writer.cpp
)

set(POLYMESH_HEADERS
    partition.hpp
    reorder.hpp
    poly_mesh.hpp
    local_mesh.hpp
    mesh_partition_manager.hpp
    mesh_quality.hpp
    vtk_writer.hpp
)

# Create library (STATIC or SHARED based on FVM_LIBRARY_TYPE)
add_library(polymesh ${FVM_LIBRARY_TYPE} ${POLYMESH_SOURCES} ${POLYMESH_HEADERS})

# Set library properties
target_include_directories(polymesh
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${Gmsh_INCLUDE_DIR}
)

# Link dependencies
# polymesh depends on meshgen for shared types (mesh_types.hpp)
target_link_libraries(polymesh
    PUBLIC
        FVM::meshgen
)

# Link METIS if available (for graph partitioning)
if(HAVE_METIS)
    target_link_libraries(polymesh PRIVATE METIS::METIS)
    target_compile_definitions(polymesh PRIVATE USE_METIS)
endif()

# Handle export definitions for shared/static builds
if(BUILD_SHARED)
    # Building shared library - define EXPORTS macro
    target_compile_definitions(polymesh PRIVATE FVM_EXPORTS)

    # Set shared library properties
    set_target_properties(polymesh PROPERTIES
        # Version information
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
        # Windows: don't add "lib" prefix
        PREFIX ""
        # Enable visibility on Unix
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
    )

    # On Windows, ensure runtime library is copied to output directory
    if(WIN32)
        set_target_properties(polymesh PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
        )
    endif()
else()
    # Building static library - define STATIC macro
    target_compile_definitions(polymesh PUBLIC FVM_STATIC)
endif()

# Set C++ standard
target_compile_features(polymesh PUBLIC cxx_std_17)

# Export alias for easier usage
add_library(FVM::polymesh ALIAS polymesh)
