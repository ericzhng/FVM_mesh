# Add library subdirectories
add_subdirectory(mesh_generation)
add_subdirectory(mesh_processing)

# Main executable
add_executable(${PROJECT_NAME} main.cpp)

# Include directories for main executable
# This allows includes like "mesh_generation/geometry.hpp"
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Gmsh_INCLUDE_DIR}
)

# Link the libraries
target_link_libraries(${PROJECT_NAME}
    PRIVATE
        FVM::mesh_generation
        FVM::mesh_processing
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

# Installation
install(TARGETS ${PROJECT_NAME} DESTINATION bin)
install(TARGETS mesh_generation mesh_processing
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
install(DIRECTORY mesh_generation/ DESTINATION include/mesh_generation
    FILES_MATCHING PATTERN "*.hpp")
install(DIRECTORY mesh_processing/ DESTINATION include/mesh_processing
    FILES_MATCHING PATTERN "*.hpp")
