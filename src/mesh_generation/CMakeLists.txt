# Mesh Generation Library - Gmsh-based geometry and mesh generation

set(MESH_GENERATION_SOURCES
    geometry.cpp
    mesh_generator.cpp
)

set(MESH_GENERATION_HEADERS
    geometry.hpp
    mesh_generator.hpp
    mesh_types.hpp
    mesh_generation_export.hpp
)

# Create library (STATIC or SHARED based on FVM_LIBRARY_TYPE)
add_library(mesh_generation ${FVM_LIBRARY_TYPE} ${MESH_GENERATION_SOURCES} ${MESH_GENERATION_HEADERS})

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

# Handle export definitions for shared/static builds
if(BUILD_SHARED)
    # Building shared library - define EXPORTS macro
    target_compile_definitions(mesh_generation PRIVATE MESH_GENERATION_EXPORTS)

    # Set shared library properties
    set_target_properties(mesh_generation PROPERTIES
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
        set_target_properties(mesh_generation PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
        )
    endif()
else()
    # Building static library - define STATIC macro
    target_compile_definitions(mesh_generation PUBLIC MESH_GENERATION_STATIC)
endif()

# Set C++ standard
target_compile_features(mesh_generation PUBLIC cxx_std_17)

# Export alias for easier usage
add_library(FVM::mesh_generation ALIAS mesh_generation)
