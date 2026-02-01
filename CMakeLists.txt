cmake_minimum_required(VERSION 3.16)
project(FVM_Mesh VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add custom CMake modules path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Option to enable METIS support
option(USE_METIS "Enable METIS library for graph partitioning" ON)

# Option to build shared libraries (DLLs on Windows, .so on Linux)
option(BUILD_SHARED "Build shared libraries instead of static" OFF)

# Set library type based on BUILD_SHARED option
if(BUILD_SHARED)
    set(FVM_LIBRARY_TYPE SHARED)
    # Enable position independent code for shared libraries
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    # On Windows, automatically export all symbols (optional, we use explicit exports)
    # set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
else()
    set(FVM_LIBRARY_TYPE STATIC)
endif()

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

# =============================================================================
# Installation Configuration
# =============================================================================
include(GNUInstallDirs)

# Set default install prefix if not specified
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Install path prefix" FORCE)
endif()

# Installation directories (set BEFORE add_subdirectory so they're available)
set(FVM_INSTALL_BINDIR ${CMAKE_INSTALL_BINDIR})
set(FVM_INSTALL_LIBDIR ${CMAKE_INSTALL_LIBDIR})
set(FVM_INSTALL_INCLUDEDIR ${CMAKE_INSTALL_INCLUDEDIR})
set(FVM_INSTALL_CMAKEDIR ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})

# Add source subdirectory
add_subdirectory(src)

# =============================================================================
# CPack Configuration - Installer Generation
# =============================================================================
set(CPACK_PACKAGE_NAME "${PROJECT_NAME}")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_VERSION_MAJOR "${PROJECT_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${PROJECT_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${PROJECT_VERSION_PATCH}")
set(CPACK_PACKAGE_VENDOR "FVM_Mesh Project")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Finite Volume Method Mesh Generation Library")
set(CPACK_PACKAGE_DESCRIPTION "A C++ library for generating and partitioning meshes for finite volume methods, featuring Gmsh integration and METIS-based partitioning.")
set(CPACK_PACKAGE_HOMEPAGE_URL "https://github.com/your-org/FVM_Mesh")
set(CPACK_PACKAGE_CONTACT "your-email@example.com")

# Optional resource files (only set if they exist)
if(EXISTS "${CMAKE_SOURCE_DIR}/LICENSE")
    set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE")
endif()
if(EXISTS "${CMAKE_SOURCE_DIR}/README.md")
    set(CPACK_RESOURCE_FILE_README "${CMAKE_SOURCE_DIR}/README.md")
endif()

# Package file name
set(CPACK_PACKAGE_FILE_NAME "${PROJECT_NAME}-${PROJECT_VERSION}-${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")

# Component-based installation
set(CPACK_COMPONENTS_ALL Runtime Development)
set(CPACK_COMPONENT_RUNTIME_DISPLAY_NAME "Runtime Libraries")
set(CPACK_COMPONENT_RUNTIME_DESCRIPTION "Shared libraries and executable needed to run FVM_Mesh applications")
set(CPACK_COMPONENT_DEVELOPMENT_DISPLAY_NAME "Development Files")
set(CPACK_COMPONENT_DEVELOPMENT_DESCRIPTION "Headers and CMake files needed to develop with FVM_Mesh")
set(CPACK_COMPONENT_DEVELOPMENT_DEPENDS Runtime)

# Platform-specific generator settings
if(WIN32)
    # NSIS installer for Windows
    set(CPACK_GENERATOR "NSIS;ZIP")

    # Find NSIS executable - check common locations and environment variable
    find_program(NSIS_EXECUTABLE
        NAMES makensis
        PATHS
            "$ENV{NSIS_ROOT}"
            "C:/Software/NSIS"
            "C:/Program Files/NSIS"
            "C:/Program Files (x86)/NSIS"
        PATH_SUFFIXES bin
        DOC "Path to NSIS makensis executable"
    )
    if(NSIS_EXECUTABLE)
        get_filename_component(NSIS_ROOT "${NSIS_EXECUTABLE}" DIRECTORY)
        set(CPACK_NSIS_EXECUTABLE "${NSIS_EXECUTABLE}")
        message(STATUS "NSIS found: ${NSIS_EXECUTABLE}")
    else()
        message(WARNING "NSIS not found. Set NSIS_ROOT environment variable or install NSIS to create installers.")
    endif()
    set(CPACK_NSIS_DISPLAY_NAME "${PROJECT_NAME} ${PROJECT_VERSION}")
    set(CPACK_NSIS_PACKAGE_NAME "${PROJECT_NAME}")
    set(CPACK_NSIS_MODIFY_PATH ON)
    set(CPACK_NSIS_ENABLE_UNINSTALL_BEFORE_INSTALL ON)

    # Optional installer icons (only set if they exist)
    if(EXISTS "${CMAKE_SOURCE_DIR}/cmake/installer.ico")
        set(CPACK_NSIS_MUI_ICON "${CMAKE_SOURCE_DIR}/cmake/installer.ico")
        set(CPACK_NSIS_MUI_UNIICON "${CMAKE_SOURCE_DIR}/cmake/installer.ico")
    endif()

    # Start menu shortcuts
    set(CPACK_NSIS_CREATE_ICONS_EXTRA
        "CreateShortCut '$SMPROGRAMS\\\\$STARTMENU_FOLDER\\\\${PROJECT_NAME}.lnk' '$INSTDIR\\\\bin\\\\${PROJECT_NAME}.exe'"
    )
    set(CPACK_NSIS_DELETE_ICONS_EXTRA
        "Delete '$SMPROGRAMS\\\\$STARTMENU_FOLDER\\\\${PROJECT_NAME}.lnk'"
    )

    # Add install directory to PATH
    set(CPACK_NSIS_EXTRA_INSTALL_COMMANDS "
        WriteRegStr HKLM 'SOFTWARE\\\\${PROJECT_NAME}' 'InstallDir' '$INSTDIR'
    ")

elseif(APPLE)
    # macOS package
    set(CPACK_GENERATOR "DragNDrop;TGZ")
    set(CPACK_DMG_VOLUME_NAME "${PROJECT_NAME}")
    set(CPACK_DMG_FORMAT "UDBZ")
    set(CPACK_BUNDLE_NAME "${PROJECT_NAME}")

elseif(UNIX)
    # Linux packages
    set(CPACK_GENERATOR "TGZ;DEB;RPM")

    # DEB specific settings
    set(CPACK_DEBIAN_PACKAGE_MAINTAINER "${CPACK_PACKAGE_CONTACT}")
    set(CPACK_DEBIAN_PACKAGE_SECTION "science")
    set(CPACK_DEBIAN_PACKAGE_PRIORITY "optional")
    set(CPACK_DEBIAN_PACKAGE_DEPENDS "libc6 (>= 2.17)")
    set(CPACK_DEBIAN_FILE_NAME DEB-DEFAULT)

    # RPM specific settings
    set(CPACK_RPM_PACKAGE_LICENSE "MIT")
    set(CPACK_RPM_PACKAGE_GROUP "Development/Libraries")
    set(CPACK_RPM_FILE_NAME RPM-DEFAULT)
endif()

# Source package configuration
set(CPACK_SOURCE_GENERATOR "TGZ;ZIP")
set(CPACK_SOURCE_IGNORE_FILES
    "/\\\\.git/"
    "/\\\\.vscode/"
    "/\\\\.idea/"
    "/build/"
    "/out/"
    "/\\\\.cache/"
    "\\\\.DS_Store"
    "~$"
)

# Include CPack module (must be after all CPACK variables are set)
include(CPack)

# =============================================================================
# Configuration Summary
# =============================================================================
message(STATUS "")
message(STATUS "=== FVM_Mesh Configuration Summary ===")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Platform: ${CMAKE_SYSTEM_NAME}")
message(STATUS "Library Type: ${FVM_LIBRARY_TYPE}")
message(STATUS "Gmsh: ${Gmsh_LIBRARIES}")
message(STATUS "METIS: ${HAVE_METIS}")
message(STATUS "")
message(STATUS "Libraries:")
message(STATUS "  - meshgen (Gmsh-based geometry and mesh generation)")
message(STATUS "  - polymesh (partition, reordering, and utilities)")
message(STATUS "")
message(STATUS "Installation:")
message(STATUS "  - Prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "  - Binaries: ${FVM_INSTALL_BINDIR}")
message(STATUS "  - Libraries: ${FVM_INSTALL_LIBDIR}")
message(STATUS "  - Headers: ${FVM_INSTALL_INCLUDEDIR}")
message(STATUS "  - CMake: ${FVM_INSTALL_CMAKEDIR}")
message(STATUS "======================================")
message(STATUS "")
