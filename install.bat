@echo off
REM UV Transfer Tool - Installation Script for Windows
REM This script automatically installs all dependencies and configures the environment

echo ============================================================
echo UV Transfer Tool - Automatic Installation Script
echo ============================================================
echo.

REM Check Python version
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH!
    echo Please install Python 3.8+ from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [1/5] Checking Python version...
python -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"
if errorlevel 1 (
    echo ERROR: Python 3.8+ is required!
    python --version
    pause
    exit /b 1
)
python --version
echo.

REM Upgrade pip
echo [2/5] Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Install core dependencies
echo [3/5] Installing core dependencies...
python -m pip install numpy scipy matplotlib Pillow pytest tqdm
echo.

REM Try to install FBX backends in order of preference
echo [4/5] Installing FBX backends...
echo.

echo Trying official FBX SDK...
python -m pip install fbx 2>nul
if errorlevel 1 (
    echo Official FBX SDK not available, trying alternatives...
    
    echo Trying pyfbx...
    python -m pip install pyfbx 2>nul
    if errorlevel 1 (
        echo pyfbx not available...
        
        echo Trying pyassimp...
        python -m pip install pyassimp 2>nul
        if errorlevel 1 (
            echo pyassimp not available...
            echo.
            echo WARNING: No external FBX backend could be installed.
            echo The native parser will be used as fallback.
            echo For best results, manually install one of:
            echo   - fbx (Official Autodesk FBX SDK)
            echo   - pyfbx
            echo   - pyassimp
        ) else (
            echo pyassimp installed successfully!
        )
    ) else (
        echo pyfbx installed successfully!
    )
) else (
    echo Official FBX SDK installed successfully!
)
echo.

REM Run environment check
echo [5/5] Verifying installation...
python "%~dp0check_env.py"
echo.

echo ============================================================
echo Installation Complete!
echo ============================================================
echo.
echo To test with your FBX files, run:
echo   python check_env.py --test-fbx "path/to/your/file.fbx"
echo.
echo To use the UV Transfer Tool, run:
echo   python -m uv_transfer --help
echo.
pause
