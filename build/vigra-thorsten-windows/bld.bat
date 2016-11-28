REM load toolset info
set TOOLSET_INFO_DIR=%PREFIX%\toolset-info
call "%TOOLSET_INFO_DIR%\common-vars.bat"

mkdir build
cd build

set CONFIGURATION=Release
set PATH=%PATH%;%LIBRARY_PREFIX%\bin

cmake .. -G "%CMAKE_GENERATOR%" -DCMAKE_PREFIX_PATH="%LIBRARY_PREFIX%" -DCMAKE_INSTALL_PREFIX="%LIBRARY_PREFIX%"
if errorlevel 1 exit 1

REM BUILD
cmake --build . --target ALL_BUILD --config %CONFIGURATION%
if errorlevel 1 exit 1

REM TEST
cmake --build . --target test_impex --config %CONFIGURATION%
if errorlevel 1 exit 1
cmake --build . --target test_hdf5impex --config %CONFIGURATION%
if errorlevel 1 exit 1
cmake --build . --target test_fourier --config %CONFIGURATION%
if errorlevel 1 exit 1
cmake --build . --target vigranumpytest --config %CONFIGURATION%
if errorlevel 1 exit 1

REM INSTALL
cmake --build . --target INSTALL --config %CONFIGURATION%
if errorlevel 1 exit 1
