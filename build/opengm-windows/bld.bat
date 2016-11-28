REM load toolset info
set TOOLSET_INFO_DIR=%PREFIX%\toolset-info
call "%TOOLSET_INFO_DIR%\common-vars-mingw.bat"

set PATH=%MSYS_PATH%;%PATH%

"%MSYS_PATH%\patch.exe" -p1 -l -N -i "%RECIPE_DIR%\opengm-external-download.patch"
REM if errorlevel 1 exit 1

mkdir build
cd build

cmake .. -G "%CMAKE_GENERATOR%" ^
         -DCMAKE_INSTALL_PREFIX="%LIBRARY_PREFIX%" ^
         -DBUILD_PYTHON_WRAPPER=ON ^
         -DPYTHON_LIBRARY="%PREFIX%\libs\python27.lib" ^
         -DPYTHON_INCLUDE_DIR="%PREFIX%\include" ^
         -DBUILD_EXAMPLES=OFF ^
         -DBUILD_TUTORIALS=OFF ^
         -DBUILD_TESTING=OFF ^
         -DWITH_BOOST=ON ^
         -DWITH_CPLEX=ON ^
         -DWITH_VIGRA=ON ^
         -DWITH_HDF5=ON
if errorlevel 1 exit 1

cmake --build . --target externalLibs
if errorlevel 1 exit 1

cmake . -DWITH_MAXFLOW=ON -DWITH_MAXFLOW_IBFS=ON -DWITH_QPBO=ON
if errorlevel 1 exit 1

cmake --build . --target ALL_BUILD --config Release
if errorlevel 1 exit 1

cmake --build . --target INSTALL --config Release
if errorlevel 1 exit 1
