@echo OFF
SETLOCAL

set INSTALL_DIR=%~dp0
call "%INSTALL_DIR%clean_paths.bat"

rem start multicut
echo Starting multicut 2D at "%INSTALL_DIR%"
"%INSTALL_DIR%python" "%INSTALL_DIR%\scripts\run_mc_2d.py" %*

ENDLOCAL
