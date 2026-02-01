/**
 * @file main.cpp
 * @brief Creates a rectangular mesh and saves it to multiple formats.
 *
 * This is the C++ equivalent of create_sample_mesh.py, generating a
 * 2D mesh suitable for use as a basis for parallel FVM solvers.
 */

#include "meshgen/geometry.hpp"
#include "meshgen/mesh_generator.hpp"
#include "meshgen/mesh_types.hpp"
#include "polymesh/vtk_writer.hpp"

#include <gmsh.h>
#include <cmath>
#include <iostream>
#include <map>
#include <string>
#include <vector>

int main(int argc, char *argv[])
{
    // Parse command line arguments (optional)
    double length = 1.0; // Length of rectangle in x-direction
    double height = 1.0; // Height of rectangle in y-direction
    double meshSize = 0.05;
    double charLength = 0.01;
    std::string outputDir = "data";
    bool showGui = false;

    std::cout << "\n Program Started !\n";
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--length" && i + 1 < argc)
        {
            length = std::stod(argv[++i]);
        }
        else if (arg == "--height" && i + 1 < argc)
        {
            height = std::stod(argv[++i]);
        }
        else if (arg == "--mesh-size" && i + 1 < argc)
        {
            meshSize = std::stod(argv[++i]);
        }
        else if (arg == "--char-length" && i + 1 < argc)
        {
            charLength = std::stod(argv[++i]);
        }
        else if (arg == "--output" && i + 1 < argc)
        {
            outputDir = argv[++i];
        }
        else if (arg == "--gui")
        {
            showGui = true;
        }
        else if (arg == "--help" || arg == "-h")
        {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  --length <value>       Rectangle length (default: 1.0)\n"
                      << "  --height <value>       Rectangle height (default: 1.0)\n"
                      << "  --mesh-size <value>    Initial mesh size (default: 0.05)\n"
                      << "  --char-length <value>  Characteristic length (default: 0.01)\n"
                      << "  --output <dir>         Output directory (default: data)\n"
                      << "  --gui                  Show Gmsh GUI after generation\n"
                      << "  --help, -h             Show this help message\n";
            return 0;
        }
    }

    std::cout << "FVM Mesh Generator - C++ Version\n";
    std::cout << "=================================\n";
    std::cout << "Parameters:\n";
    std::cout << "  Length: " << length << "\n";
    std::cout << "  Height: " << height << "\n";
    std::cout << "  Mesh size: " << meshSize << "\n";
    std::cout << "  Char length: " << charLength << "\n";
    std::cout << "  Output: " << outputDir << "\n\n";

    try
    {
        // Initialize Gmsh
        gmsh::initialize();
        gmsh::model::add("test_polygon");

        // Create geometry
        fvm::Geometry geom("Rectangular Domain");
        int surfaceTag = geom.rectangle(length, height, 0.0, 0.0, meshSize);

        // Synchronize the geo kernel
        gmsh::model::geo::synchronize();

        // Get boundary entities
        std::vector<std::pair<int, int>> boundaryEntities;
        gmsh::model::getBoundary({{2, surfaceTag}}, boundaryEntities);

        // Extract line tags
        std::vector<int> lineTags;
        for (const auto &[dim, tag] : boundaryEntities)
        {
            lineTags.push_back(std::abs(tag));
        }

        // Identify boundaries by position
        int topBc = -1;
        int bottomBc = -1;
        int leftBc = -1;
        int rightBc = -1;

        const double tol = 1e-9;

        for (int lineTag : lineTags)
        {
            double minX, minY, minZ, maxX, maxY, maxZ;
            gmsh::model::getBoundingBox(1, lineTag, minX, minY, minZ, maxX, maxY, maxZ);

            if (std::abs(minY - 0.0) < tol && std::abs(maxY - 0.0) < tol)
            {
                bottomBc = lineTag;
            }
            else if (std::abs(minX - length) < tol && std::abs(maxX - length) < tol)
            {
                rightBc = lineTag;
            }
            else if (std::abs(minY - height) < tol && std::abs(maxY - height) < tol)
            {
                topBc = lineTag;
            }
            else if (std::abs(minX - 0.0) < tol && std::abs(maxX - 0.0) < tol)
            {
                leftBc = lineTag;
            }
        }

        // Add physical groups for boundaries
        if (leftBc != -1)
        {
            gmsh::model::addPhysicalGroup(1, {leftBc}, -1, "inlet");
            std::cout << "Added boundary: inlet (line " << leftBc << ")\n";
        }
        if (rightBc != -1)
        {
            gmsh::model::addPhysicalGroup(1, {rightBc}, -1, "outlet");
            std::cout << "Added boundary: outlet (line " << rightBc << ")\n";
        }
        if (bottomBc != -1 && topBc != -1)
        {
            gmsh::model::addPhysicalGroup(1, {bottomBc, topBc}, -1, "wall");
            std::cout << "Added boundary: wall (lines " << bottomBc << ", " << topBc << ")\n";
        }

        std::cout << "\n";

        // Generate mesh
        fvm::MeshGenerator mesher(surfaceTag, outputDir);

        std::map<int, fvm::MeshParams> meshParams;
        meshParams[surfaceTag] = {fvm::MeshType::Quads, charLength};

        mesher.generate(meshParams, "sample_rect_mesh.msh");

        // Get mesh data for export
        const fvm::MeshData &meshData = mesher.getMeshData();

        // Write to VTK format (legacy)
        fvm::VTKWriter::writeVTK(meshData, outputDir + "/sample_rect_mesh.vtk");

        // Write to VTU format (XML, better for ParaView)
        fvm::VTKWriter::writeVTU(meshData, outputDir + "/sample_rect_mesh.vtu");

        // Write boundary information
        fvm::VTKWriter::writeBoundaryInfo(meshData, outputDir + "/boundary_info.txt");

        // Write OpenFOAM format (for OpenFOAM-based solvers)
        fvm::VTKWriter::writeOpenFOAM(meshData, outputDir + "/openfoam");

        std::cout << "\nMesh export complete!\n";
        std::cout << "Output files:\n";
        std::cout << "  - " << outputDir << "/sample_rect_mesh.msh (Gmsh format)\n";
        std::cout << "  - " << outputDir << "/sample_rect_mesh.vtk (VTK legacy format)\n";
        std::cout << "  - " << outputDir << "/sample_rect_mesh.vtu (VTK XML format)\n";
        std::cout << "  - " << outputDir << "/boundary_info.txt (boundary conditions)\n";
        std::cout << "  - " << outputDir << "/openfoam/constant/polyMesh/ (OpenFOAM format)\n";

        // Show GUI if requested
        if (showGui)
        {
            std::cout << "\nOpening Gmsh GUI...\n";
            gmsh::fltk::run();
        }

        // Finalize Gmsh
        gmsh::finalize();

        std::cout << "\nDone!\n";
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        gmsh::finalize();
        return 1;
    }
}
