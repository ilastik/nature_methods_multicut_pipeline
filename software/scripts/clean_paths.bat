rem find CPLEX in the original PATH. FIXME: do we need to be more specific, e.g. cplex125.dll ?
where /Q cplex
if NOT ERRORLEVEL 1 for /f "delims=" %%a in ('where cplex') do @set CPLEX_PATH=%%~dpa
if NOT DEFINED CPLEX_PATH (
    echo ##########################################################################
    echo #########            CPLEX LIBRARY HAS NOT BEEN FOUND!!!           #######
    echo ##########################################################################
    echo ######### you have cplex? make sure it is in the PATH!             #######
    echo ##########################################################################
    echo ######### don't have cplex? apply for an academic license at IBM!  #######
    echo #########               see README.txt for details                 #######
    echo ##########################################################################
    exit
)

rem overwrite PATH with only what multicut needs
set INSTALL_DIR=%~dp0
set PATH=%INSTALL_DIR%;%INSTALL_DIR%Library\bin;%INSTALL_DIR%DLLs

rem re-insert CPLEX into the PATH
if DEFINED CPLEX_PATH set PATH=%PATH%;%CPLEX_PATH%

rem set more paths
set PYTHONHOME=%INSTALL_DIR%
set PYTHONNOUSERSITE=1
