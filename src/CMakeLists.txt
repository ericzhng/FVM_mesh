# Source files
set(SOURCES
    geometry.cpp
    mesh_generator.cpp
    vtk_writer.cpp
    poly_mesh.cpp
    mesh_quality.cpp
    reorder.cpp
    partition.cpp
    local_mesh.cpp
    mesh_partition_manager.cpp
    main.cpp
)

# Header files
set(HEADERS
    geometry.hpp
    mesh_generator.hpp
    vtk_writer.hpp
    mesh_types.hpp
    poly_mesh.hpp
    mesh_quality.hpp
    reorder.hpp
    partition.hpp
    local_mesh.hpp
    mesh_partition_manager.hpp
)

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Include directories
# IMPORTANT: Gmsh C wrapper include must come BEFORE vcpkg includes
# to override vcpkg's gmsh.h with our C API wrapper (gmsh.h_cwrap)
target_include_directories(${PROJECT_NAME} BEFORE PRIVATE
    ${Gmsh_INCLUDE_DIR}
)
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link Gmsh (required)
target_link_libraries(${PROJECT_NAME} PRIVATE Gmsh::Gmsh)
target_compile_definitions(${PROJECT_NAME} PRIVATE GMSH_DLL)

# Copy Gmsh DLL to output directory after build
if(WIN32 AND GMSH_DLL)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${GMSH_DLL}"
            "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
        COMMENT "Copying Gmsh DLL to output directory"
    )
endif()

# Link METIS if available
if(HAVE_METIS)
    target_link_libraries(${PROJECT_NAME} PRIVATE METIS::METIS)
    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_METIS)
endif()

# Installation
install(TARGETS ${PROJECT_NAME} DESTINATION bin)
