@echo off
REM Generate MSVC import library from gmsh DLL
REM Run this from VS Developer Command Prompt

set GMSH_DIR=C:\Users\ZhangHui\.conda\envs\env_api\Lib
set OUTPUT_DIR=C:\Users\ZhangHui\.conda\envs\env_api\Lib

REM Export symbols from DLL
dumpbin /EXPORTS "%GMSH_DIR%\gmsh-4.15.dll" > "%OUTPUT_DIR%\gmsh_exports.txt"

REM Create .def file
echo LIBRARY gmsh-4.15 > "%OUTPUT_DIR%\gmsh.def"
echo EXPORTS >> "%OUTPUT_DIR%\gmsh.def"

REM Parse exports and add to def file (you may need to manually edit this)
for /f "tokens=4" %%a in ('dumpbin /EXPORTS "%GMSH_DIR%\gmsh-4.15.dll" ^| findstr /R "^[ ]*[0-9]"') do (
    echo %%a >> "%OUTPUT_DIR%\gmsh.def"
)

REM Generate import library
lib /def:"%OUTPUT_DIR%\gmsh.def" /out:"%OUTPUT_DIR%\gmsh_msvc.lib" /machine:x64

echo Done! Use gmsh_msvc.lib instead of gmsh.dll.lib
