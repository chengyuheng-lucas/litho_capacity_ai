@echo off
setlocal enableextensions

cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python not found. Please install Python 3.12+ and ensure it is on PATH.
  exit /b 1
)

echo [INFO] Python:
python -V

echo [INFO] Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 exit /b 1

echo [INFO] Installing PyTorch (CPU)...
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 exit /b 1

echo [INFO] Installing dependencies...
python -m pip install -r requirements.txt
if errorlevel 1 exit /b 1

echo [INFO] Installing project (editable)...
python -m pip install -e .
if errorlevel 1 exit /b 1

echo [OK] Done.
exit /b 0
