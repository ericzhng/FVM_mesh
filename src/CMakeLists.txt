# Add library subdirectories
add_subdirectory(meshgen)
add_subdirectory(polymesh)

# Main executable
add_executable(${PROJECT_NAME} main.cpp)

# Include directories for main executable
# This allows includes like "meshgen/geometry.hpp"
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Gmsh_INCLUDE_DIR}
)

# Link the libraries
target_link_libraries(${PROJECT_NAME}
    PRIVATE
        FVM::meshgen
        FVM::polymesh
)

# Copy Gmsh DLL to output directory after build (Windows only)
if(WIN32 AND GMSH_DLL)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${GMSH_DLL}"
            "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
        COMMENT "Copying Gmsh DLL to output directory"
    )
endif()

# Copy METIS DLL to output directory after build (Windows only)
if(WIN32 AND HAVE_METIS AND METIS_DLL)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${METIS_DLL}"
            "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
        COMMENT "Copying METIS DLL to output directory"
    )
endif()

# Copy our library DLLs to executable directory (Windows shared builds only)
if(WIN32 AND BUILD_SHARED)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:meshgen>"
            "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
        COMMENT "Copying meshgen DLL to output directory"
    )
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:polymesh>"
            "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
        COMMENT "Copying polymesh DLL to output directory"
    )
endif()

# =============================================================================
# Installation Rules
# =============================================================================

# Install the main executable
install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION ${FVM_INSTALL_BINDIR}
    COMPONENT Runtime
)

# Install libraries with proper export configuration
install(TARGETS meshgen polymesh
    EXPORT FVM_meshTargets
    ARCHIVE DESTINATION ${FVM_INSTALL_LIBDIR} COMPONENT Development
    LIBRARY DESTINATION ${FVM_INSTALL_LIBDIR} COMPONENT Runtime
    RUNTIME DESTINATION ${FVM_INSTALL_BINDIR} COMPONENT Runtime  # For Windows DLLs
    INCLUDES DESTINATION ${FVM_INSTALL_INCLUDEDIR}
)

# Install header files
install(FILES fvm_export.hpp
    DESTINATION ${FVM_INSTALL_INCLUDEDIR}
    COMPONENT Development
)
install(DIRECTORY meshgen/
    DESTINATION ${FVM_INSTALL_INCLUDEDIR}/meshgen
    COMPONENT Development
    FILES_MATCHING PATTERN "*.hpp"
)
install(DIRECTORY polymesh/
    DESTINATION ${FVM_INSTALL_INCLUDEDIR}/polymesh
    COMPONENT Development
    FILES_MATCHING PATTERN "*.hpp"
)

# Export targets for use by other projects
install(EXPORT FVM_meshTargets
    FILE FVM_meshTargets.cmake
    NAMESPACE FVM::
    DESTINATION ${FVM_INSTALL_CMAKEDIR}
    COMPONENT Development
)

# Generate package version file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/FVM_meshConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

# Generate package config file from template
configure_package_config_file(
    "${CMAKE_SOURCE_DIR}/cmake/FVM_meshConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/FVM_meshConfig.cmake"
    INSTALL_DESTINATION ${FVM_INSTALL_CMAKEDIR}
)

# Install CMake configuration files
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/FVM_meshConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/FVM_meshConfigVersion.cmake"
    DESTINATION ${FVM_INSTALL_CMAKEDIR}
    COMPONENT Development
)

# Install FindGmsh and FindMETIS for consumers who might need them
install(FILES
    "${CMAKE_SOURCE_DIR}/cmake/FindGmsh.cmake"
    "${CMAKE_SOURCE_DIR}/cmake/FindMETIS.cmake"
    DESTINATION ${FVM_INSTALL_CMAKEDIR}/modules
    COMPONENT Development
)

# Copy Gmsh DLL to install bin directory (Windows only)
if(WIN32 AND GMSH_DLL)
    install(FILES "${GMSH_DLL}"
        DESTINATION ${FVM_INSTALL_BINDIR}
        COMPONENT Runtime
    )
endif()

# Copy METIS DLL if available (Windows only)
if(WIN32 AND HAVE_METIS AND METIS_DLL)
    install(FILES "${METIS_DLL}"
        DESTINATION ${FVM_INSTALL_BINDIR}
        COMPONENT Runtime
    )
endif()
