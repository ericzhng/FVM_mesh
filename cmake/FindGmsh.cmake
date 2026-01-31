# FindGmsh.cmake
# Finds Gmsh from bundled SDK (converted from pip for MSVC compatibility)
# The bundled SDK uses the C API wrapper header for cross-compiler compatibility
cmake_minimum_required(VERSION 3.16)

# Bundled SDK path (preferred)
set(GMSH_BUNDLED_DIR "${CMAKE_SOURCE_DIR}/extern/gmsh-4.15.0")

# Try bundled SDK first (recommended), then conda environment
if(EXISTS "${GMSH_BUNDLED_DIR}/include/gmsh.h")
    # Bundled SDK with pre-converted MSVC-compatible library
    set(Gmsh_INCLUDE_DIR "${GMSH_BUNDLED_DIR}/include")
    set(Gmsh_LIBRARY "${GMSH_BUNDLED_DIR}/lib/gmsh_msvc.lib")
    set(GMSH_DLL "${GMSH_BUNDLED_DIR}/dll/gmsh-4.15.dll")
    message(STATUS "Using bundled Gmsh SDK: ${GMSH_BUNDLED_DIR}")
else()
    message(FATAL_ERROR "Gmsh not found! Please ensure extern/gmsh-4.15.0 exists with include/, lib/, and dll/ subdirectories.")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Gmsh
    REQUIRED_VARS Gmsh_LIBRARY Gmsh_INCLUDE_DIR
)

if(Gmsh_FOUND)
    set(Gmsh_LIBRARIES ${Gmsh_LIBRARY})
    set(Gmsh_INCLUDE_DIRS ${Gmsh_INCLUDE_DIR})

    if(NOT TARGET Gmsh::Gmsh)
        add_library(Gmsh::Gmsh SHARED IMPORTED)
        set_target_properties(Gmsh::Gmsh PROPERTIES
            IMPORTED_IMPLIB "${Gmsh_LIBRARY}"
            IMPORTED_LOCATION "${GMSH_DLL}"
            # NOTE: We don't set INTERFACE_INCLUDE_DIRECTORIES here because
            # we need to ensure our C wrapper gmsh.h comes BEFORE vcpkg's gmsh.h.
            # The include directory is added explicitly in src/CMakeLists.txt.
        )
    endif()
endif()
