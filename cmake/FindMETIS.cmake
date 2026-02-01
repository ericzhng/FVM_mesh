# FindMETIS.cmake
# Finds METIS library installed via vcpkg
cmake_minimum_required(VERSION 3.16)

# Try to use VCPKG_ROOT environment variable, fallback to hardcoded path
if(DEFINED ENV{VCPKG_ROOT})
    set(METIS_ROOT "$ENV{VCPKG_ROOT}/installed/x64-windows")
else()
    set(METIS_ROOT "C:/dev/vcpkg/installed/x64-windows")
endif()

# Find include directory
find_path(METIS_INCLUDE_DIR
    NAMES metis.h
    PATHS "${METIS_ROOT}/include"
    NO_DEFAULT_PATH
)

# Find libraries based on build type
if(CMAKE_BUILD_TYPE MATCHES "Debug" OR NOT CMAKE_BUILD_TYPE)
    # Debug libraries
    find_library(METIS_LIBRARY
        NAMES metis
        PATHS "${METIS_ROOT}/debug/lib"
        NO_DEFAULT_PATH
    )
    find_library(GK_LIBRARY
        NAMES gklib GKlib
        PATHS "${METIS_ROOT}/debug/lib"
        NO_DEFAULT_PATH
    )
    # Debug DLLs
    find_file(METIS_DLL
        NAMES metis.dll
        PATHS "${METIS_ROOT}/debug/bin"
        NO_DEFAULT_PATH
    )
else()
    # Release libraries
    find_library(METIS_LIBRARY
        NAMES metis
        PATHS "${METIS_ROOT}/lib"
        NO_DEFAULT_PATH
    )
    find_library(GK_LIBRARY
        NAMES gklib GKlib
        PATHS "${METIS_ROOT}/lib"
        NO_DEFAULT_PATH
    )
    # Release DLLs
    find_file(METIS_DLL
        NAMES metis.dll
        PATHS "${METIS_ROOT}/bin"
        NO_DEFAULT_PATH
    )
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(METIS
    REQUIRED_VARS METIS_LIBRARY METIS_INCLUDE_DIR GK_LIBRARY
)

if(METIS_FOUND)
    set(METIS_LIBRARIES ${METIS_LIBRARY} ${GK_LIBRARY})
    set(METIS_INCLUDE_DIRS ${METIS_INCLUDE_DIR})

    if(NOT TARGET METIS::METIS)
        add_library(METIS::METIS INTERFACE IMPORTED)
        set_target_properties(METIS::METIS PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${METIS_INCLUDE_DIRS}"
            INTERFACE_LINK_LIBRARIES "${METIS_LIBRARIES}"
        )
    endif()

    # Report DLL status
    if(METIS_DLL)
        message(STATUS "METIS DLL found: ${METIS_DLL}")
    else()
        message(STATUS "METIS DLL not found (static linking or manual copy required)")
    endif()
endif()
