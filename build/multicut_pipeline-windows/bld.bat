set PARENT_DIR=%RECIPE_DIR%\..\..
copy "%PARENT_DIR%\software\multicut_src\*.py" "%PREFIX%\Lib\site-packages"
mkdir "%PREFIX%\scripts"
copy "%PARENT_DIR%\software\scripts\*.py" "%PREFIX%\scripts"
copy "%PARENT_DIR%\software\scripts\*.bat" "%PREFIX%"
copy "%PARENT_DIR%\software\scripts\README-WIN.txt" "%PREFIX%"
