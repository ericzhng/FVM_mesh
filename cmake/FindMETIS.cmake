# FindMETIS.cmake
cmake_minimum_required(VERSION 3.16)

set(METIS_ROOT "C:/dev/vcpkg/installed/x64-windows")

set(METIS_INCLUDE_DIR "${METIS_ROOT}/include")
# Note: This is linking the debug library. For a release build, you'd want the release version.
set(METIS_LIBRARY "${METIS_ROOT}/debug/lib/metis.lib")
set(GK_LIBRARY "${METIS_ROOT}/debug/lib/gklib.lib")

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
endif()
