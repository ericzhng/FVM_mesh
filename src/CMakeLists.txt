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

# Installation
install(TARGETS ${PROJECT_NAME} DESTINATION bin)
install(TARGETS meshgen polymesh
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin  # For Windows DLLs
)
install(FILES fvm_export.hpp DESTINATION include)
install(DIRECTORY meshgen/ DESTINATION include/meshgen
    FILES_MATCHING PATTERN "*.hpp")
install(DIRECTORY polymesh/ DESTINATION include/polymesh
    FILES_MATCHING PATTERN "*.hpp")
