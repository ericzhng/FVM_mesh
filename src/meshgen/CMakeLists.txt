# Meshgen Library - Gmsh-based geometry and mesh generation

set(MESHGEN_SOURCES
    geometry.cpp
    mesh_generator.cpp
)

set(MESHGEN_HEADERS
    geometry.hpp
    mesh_generator.hpp
    mesh_types.hpp
)

# Create library (STATIC or SHARED based on FVM_LIBRARY_TYPE)
add_library(meshgen ${FVM_LIBRARY_TYPE} ${MESHGEN_SOURCES} ${MESHGEN_HEADERS})

# Set library properties
target_include_directories(meshgen
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${Gmsh_INCLUDE_DIR}
)

# Link Gmsh (required for this library)
target_link_libraries(meshgen PUBLIC Gmsh::Gmsh)
target_compile_definitions(meshgen PUBLIC GMSH_DLL)

# Handle export definitions for shared/static builds
if(BUILD_SHARED)
    # Building shared library - define EXPORTS macro
    target_compile_definitions(meshgen PRIVATE FVM_EXPORTS)

    # Set shared library properties
    set_target_properties(meshgen PROPERTIES
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
        set_target_properties(meshgen PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
        )
    endif()
else()
    # Building static library - define STATIC macro
    target_compile_definitions(meshgen PUBLIC FVM_STATIC)
endif()

# Set C++ standard
target_compile_features(meshgen PUBLIC cxx_std_17)

# Export alias for easier usage
add_library(FVM::meshgen ALIAS meshgen)
