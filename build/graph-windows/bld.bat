REM load toolset info
set TOOLSET_INFO_DIR=%PREFIX%\toolset-info
call "%TOOLSET_INFO_DIR%\common-vars.bat"

cd dependencies\graph-1.6
mkdir build
cd build

cmake .. -G "%CMAKE_GENERATOR%" ^
         -DCMAKE_INSTALL_PREFIX="%LIBRARY_PREFIX%" ^
         -DCMAKE_PREFIX_PATH="%LIBRARY_PREFIX%" ^
         -DBUILD_PYTHON=ON ^
         -DPYTHON_LIBRARY="%PREFIX%\libs\python27.lib" ^
         -DPYTHON_INCLUDE_DIR="%PREFIX%\include" ^
         -DPYTHON_NUMPY_INCLUDE_DIR="%PREFIX%\Lib\site-packages\numpy\core\include" ^
         -DVIGRA_INCLUDE_DIR="%LIBRARY_PREFIX%\include"
if errorlevel 1 exit 1

cmake --build . --target ALL_BUILD --config Release
if errorlevel 1 exit 1

set INSTALL_DIR=%PREFIX%\Lib\site-packages\graph
mkdir "%INSTALL_DIR%"
copy ..\src\andres\graph\python\module\__init__.py "%INSTALL_DIR%\__init__.py"
copy Release\_graph.so "%INSTALL_DIR%\_graph.pyd"
