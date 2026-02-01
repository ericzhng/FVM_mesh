/**
 * @file fvm_export.hpp
 * @brief Unified export/import macros for FVM libraries
 *
 * This header provides cross-platform support for building shared libraries.
 * On Windows, it handles __declspec(dllexport/dllimport).
 * On other platforms, it uses visibility attributes for optimization.
 */

#ifndef FVM_EXPORT_HPP
#define FVM_EXPORT_HPP

// =============================================================================
// FVM Library Export Macros
// =============================================================================

#if defined(_WIN32) || defined(_WIN64)
    // Windows platform
    #if defined(FVM_STATIC)
        // Static library - no export/import needed
        #define FVM_API
    #elif defined(FVM_EXPORTS)
        // Building the DLL - export symbols
        #define FVM_API __declspec(dllexport)
    #else
        // Using the DLL - import symbols
        #define FVM_API __declspec(dllimport)
    #endif
#else
    // Non-Windows platforms (Linux, macOS, etc.)
    #if defined(FVM_EXPORTS) && defined(__GNUC__) && __GNUC__ >= 4
        // Use visibility attribute for GCC/Clang
        #define FVM_API __attribute__((visibility("default")))
    #else
        #define FVM_API
    #endif
#endif

// =============================================================================
// Deprecation Warning Macros
// =============================================================================

#if defined(_MSC_VER)
    #define FVM_DEPRECATED __declspec(deprecated)
#elif defined(__GNUC__) || defined(__clang__)
    #define FVM_DEPRECATED __attribute__((deprecated))
#else
    #define FVM_DEPRECATED
#endif

#endif // FVM_EXPORT_HPP
