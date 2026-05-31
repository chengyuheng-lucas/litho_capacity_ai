@echo off
setlocal enableextensions enabledelayedexpansion

cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python not found. Please install Python 3.12+ and ensure it is on PATH.
  exit /b 1
)

:MENU
echo.
echo ============================================
echo Litho Capacity AI - Interactive Runner
echo ============================================
echo 1) Train (public, stable baseline - FRED fast)
echo 2) Train (simulated)
echo 3) Infer (public)
echo 4) Infer (simulated)
echo 5) Exit
echo.
set /p CHOICE=Select an option (1-5):

if "%CHOICE%"=="1" goto TRAIN_PUBLIC
if "%CHOICE%"=="2" goto TRAIN_SIM
if "%CHOICE%"=="3" goto INFER_PUBLIC
if "%CHOICE%"=="4" goto INFER_SIM
if "%CHOICE%"=="5" goto END

echo [WARN] Invalid selection.
goto MENU

:TRAIN_PUBLIC
set /p OUT_DIR=Output dir [artifacts_public]:
if "%OUT_DIR%"=="" set OUT_DIR=artifacts_public

set /p START=Start date (YYYY-MM-DD) [2020-01-01]:
if "%START%"=="" set START=2020-01-01

set /p EPOCHS=Epochs [15]:
if "%EPOCHS%"=="" set EPOCHS=15

set /p BATCH=Batch size [64]:
if "%BATCH%"=="" set BATCH=64

set /p TIMEOUT=HTTP timeout seconds [10]:
if "%TIMEOUT%"=="" set TIMEOUT=10

set /p RETRIES=HTTP retries [1]:
if "%RETRIES%"=="" set RETRIES=1

echo.
echo [INFO] Training public model...
python -m litho_capacity_ai.train --data public --public_profile fred_fast --out_dir "%OUT_DIR%" --start "%START%" --epochs %EPOCHS% --batch_size %BATCH% --timeout_s %TIMEOUT% --retries %RETRIES%
echo.
pause
goto MENU

:TRAIN_SIM
set /p OUT_DIR=Output dir [artifacts]:
if "%OUT_DIR%"=="" set OUT_DIR=artifacts

set /p EPOCHS=Epochs [3]:
if "%EPOCHS%"=="" set EPOCHS=3

echo.
echo [INFO] Training simulated model...
python -m litho_capacity_ai.train --data simulated --out_dir "%OUT_DIR%" --epochs %EPOCHS%
echo.
pause
goto MENU

:INFER_PUBLIC
set /p MODEL_DIR=Model dir [artifacts_public]:
if "%MODEL_DIR%"=="" set MODEL_DIR=artifacts_public

set /p TIMEOUT=HTTP timeout seconds [10]:
if "%TIMEOUT%"=="" set TIMEOUT=10

set /p RETRIES=HTTP retries [1]:
if "%RETRIES%"=="" set RETRIES=1

echo.
echo [INFO] Running public inference...
python -m litho_capacity_ai.infer --model_dir "%MODEL_DIR%" --data public --timeout_s %TIMEOUT% --retries %RETRIES%
echo.
pause
goto MENU

:INFER_SIM
set /p MODEL_DIR=Model dir [artifacts]:
if "%MODEL_DIR%"=="" set MODEL_DIR=artifacts

echo.
echo [INFO] Running simulated inference...
python -m litho_capacity_ai.infer --model_dir "%MODEL_DIR%" --data simulated
echo.
pause
goto MENU

:END
echo [OK] Bye.
exit /b 0

