cmake_minimum_required(VERSION 3.16)
project(FVM_Mesh VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add custom CMake modules path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Option to enable METIS support
option(USE_METIS "Enable METIS library for graph partitioning" ON)

# =============================================================================
# Find Dependencies
# =============================================================================
# Gmsh SDK is bundled in extern/gmsh-4.15.0
# METIS uses custom FindMETIS.cmake or environment variables
# =============================================================================

# Find Gmsh (required) - uses FindGmsh.cmake
find_package(Gmsh REQUIRED)
message(STATUS "Gmsh found: ${Gmsh_LIBRARIES}")

# Find METIS (optional)
if(USE_METIS)
    find_package(METIS)
    if(METIS_FOUND)
        message(STATUS "METIS found: ${METIS_LIBRARIES}")
        set(HAVE_METIS TRUE)
    else()
        message(WARNING "METIS not found. Partitioning will fall back to hierarchical method.")
        set(HAVE_METIS FALSE)
    endif()
else()
    set(HAVE_METIS FALSE)
endif()

# Create data output directory
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/data)

# Add source subdirectory
add_subdirectory(src)

# =============================================================================
# Configuration Summary
# =============================================================================
message(STATUS "")
message(STATUS "=== FVM_Mesh Configuration Summary ===")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Platform: ${CMAKE_SYSTEM_NAME}")
message(STATUS "Gmsh: ${Gmsh_LIBRARIES}")
message(STATUS "METIS: ${HAVE_METIS}")
message(STATUS "")
message(STATUS "Libraries:")
message(STATUS "  - mesh_generation (Gmsh-based geometry and mesh generation)")
message(STATUS "  - mesh_processing (partition, reordering, and utilities)")
message(STATUS "======================================")
message(STATUS "")
