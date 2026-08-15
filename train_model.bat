@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "DATASET=%~1"
if not defined DATASET set "DATASET=%CD%\data\dataset\dataset.zip"

if not exist "%DATASET%" (
  echo.
  echo [ERROR] Dataset not found:
  echo   %DATASET%
  echo.
  echo Put your dataset here:
  echo   %CD%\data\dataset\dataset.zip
  echo.
  echo Then run train_model.bat again.
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  py -3.12 -m venv .venv 2>nul || py -3.11 -m venv .venv 2>nul || python -m venv .venv
  if errorlevel 1 exit /b 1
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 exit /b 1

echo.
echo === Checking dataset ===
python training\inspect_dataset.py --data "%DATASET%"
if errorlevel 1 exit /b 1

echo.
echo === Training EfficientNet-B0 ===
python training\train.py --data "%DATASET%" --arch efficientnet_b0 --epochs 30 --batch-size 16 --lr 3e-4 --device auto --workers 0 --amp
if errorlevel 1 exit /b 1

echo.
echo Training finished.
echo Best model: %CD%\models\best_model.pt
endlocal
