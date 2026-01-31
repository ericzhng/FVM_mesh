# Mesh Generation Library - Gmsh-based geometry and mesh generation

set(MESH_GENERATION_SOURCES
    geometry.cpp
    mesh_generator.cpp
)

set(MESH_GENERATION_HEADERS
    geometry.hpp
    mesh_generator.hpp
    mesh_types.hpp
)

# Create static library
add_library(mesh_generation STATIC ${MESH_GENERATION_SOURCES} ${MESH_GENERATION_HEADERS})

# Set library properties
target_include_directories(mesh_generation
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include/mesh_generation>
    PRIVATE
        ${Gmsh_INCLUDE_DIR}
)

# Link Gmsh (required for this library)
target_link_libraries(mesh_generation PUBLIC Gmsh::Gmsh)
target_compile_definitions(mesh_generation PUBLIC GMSH_DLL)

# Set C++ standard
target_compile_features(mesh_generation PUBLIC cxx_std_17)

# Export alias for easier usage
add_library(FVM::mesh_generation ALIAS mesh_generation)
