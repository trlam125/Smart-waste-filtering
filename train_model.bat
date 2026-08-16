@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "DATASET=%~1"
if not defined DATASET set "DATASET=%CD%\data\dataset\dataset.zip"

set "OUTPUT=%CD%\runs\efficientnet_b0"
set "CHECKPOINT=%OUTPUT%\last_checkpoint.pt"

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
if exist "%CHECKPOINT%" (
  echo === Resume checkpoint found ===
  echo   %CHECKPOINT%
  echo Training will continue from the last completed epoch.
  echo.
  echo === Training EfficientNet-B0 - RESUME ===
  python training\train.py --data "%DATASET%" --arch efficientnet_b0 --image-size 224 --epochs 40 --batch-size 16 --lr 3e-4 --weight-decay 1e-4 --label-smoothing 0.08 --class-weighting sqrt --patience 8 --device auto --workers 0 --amp --output "%OUTPUT%" --resume "%CHECKPOINT%"
) else (
  echo === No resume checkpoint found ===
  echo Training will start from epoch 1.
  echo.
  echo === Training EfficientNet-B0 - NEW ===
  python training\train.py --data "%DATASET%" --arch efficientnet_b0 --image-size 224 --epochs 40 --batch-size 16 --lr 3e-4 --weight-decay 1e-4 --label-smoothing 0.08 --class-weighting sqrt --patience 8 --device auto --workers 0 --amp --output "%OUTPUT%"
)

if errorlevel 1 (
  echo.
  echo Training stopped or failed.
  if exist "%CHECKPOINT%" (
    echo Resume checkpoint kept at:
    echo   %CHECKPOINT%
    echo Run train_model.bat again to continue.
  ) else (
    echo No completed-epoch checkpoint is available yet.
  )
  exit /b 1
)

echo.
echo Training finished successfully.

if exist "%CHECKPOINT%" (
  del /q "%CHECKPOINT%"
  echo Resume checkpoint removed because training completed successfully.
)

echo Best model: %CD%\models\best_model.pt
endlocal
