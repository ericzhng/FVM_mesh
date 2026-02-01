/**
 * @file mesh_processing_export.hpp
 * @brief Export/import macros for mesh_processing library
 *
 * This header provides cross-platform support for building shared libraries.
 * On Windows, it handles __declspec(dllexport/dllimport).
 * On other platforms, it uses visibility attributes for optimization.
 */

#ifndef MESH_PROCESSING_EXPORT_HPP
#define MESH_PROCESSING_EXPORT_HPP

// Determine the appropriate export/import macros based on platform and build type
#if defined(_WIN32) || defined(_WIN64)
    // Windows platform
    #ifdef MESH_PROCESSING_STATIC
        // Static library - no export/import needed
        #define MESH_PROCESSING_API
    #elif defined(MESH_PROCESSING_EXPORTS)
        // Building the DLL - export symbols
        #define MESH_PROCESSING_API __declspec(dllexport)
    #else
        // Using the DLL - import symbols
        #define MESH_PROCESSING_API __declspec(dllimport)
    #endif
#else
    // Non-Windows platforms (Linux, macOS, etc.)
    #if defined(MESH_PROCESSING_EXPORTS) && defined(__GNUC__) && __GNUC__ >= 4
        // Use visibility attribute for GCC/Clang
        #define MESH_PROCESSING_API __attribute__((visibility("default")))
    #else
        #define MESH_PROCESSING_API
    #endif
#endif

// Helper macro for deprecation warnings
#if defined(_MSC_VER)
    #define MESH_PROCESSING_DEPRECATED __declspec(deprecated)
#elif defined(__GNUC__) || defined(__clang__)
    #define MESH_PROCESSING_DEPRECATED __attribute__((deprecated))
#else
    #define MESH_PROCESSING_DEPRECATED
#endif

#endif // MESH_PROCESSING_EXPORT_HPP
