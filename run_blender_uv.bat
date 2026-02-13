@echo off
REM Run Blender UV visualization script

echo ============================================
echo Blender UV Visualization
echo ============================================
echo.

REM Check if Blender is in PATH
where blender >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Blender not found in PATH!
    echo Please install Blender and add it to your PATH environment variable.
    echo.
    echo Or specify the full path to blender.exe:
    echo   run_blender_uv.bat "C:\Program Files\Blender Foundation\Blender 3.6\blender.exe"
    pause
    exit /b 1
)

set BLENDER_EXE=blender
if not "%~1"=="" (
    set BLENDER_EXE=%~1
)

echo Using Blender: %BLENDER_EXE%
echo.

REM Run the Python script in Blender
%BLENDER_EXE% --background --python "%~dp0test_blender_uv.py"

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Blender script failed!
    pause
    exit /b 1
)

echo.
echo ============================================
echo Blender UV visualization completed!
echo Check the output directory for results.
echo ============================================
pause
