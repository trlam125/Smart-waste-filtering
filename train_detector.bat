@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

set "MODE=%~1"
set "DATASET=%~2"
if not defined DATASET set "DATASET=%CD%\data\dataset\detector_class\ObjectDetector_dataset.zip"

set "MODEL=yolo26s.pt"
set "EPOCHS=100"
set "IMGSZ=640"
set "BATCH=8"
set "WORKERS=0"
set "PATIENCE=20"
set "SEED=42"
set "RUN_NAME=smartwaste_detector"
set "EXTRA_ARGS="

if /I "%MODE%"=="test" (
  set "EPOCHS=3"
  set "RUN_NAME=smartwaste_detector_test"
  set "EXTRA_ARGS=--no-deploy"
) else if not "%MODE%"=="" (
  if /I not "%MODE%"=="train" (
    echo Usage:
    echo   train_detector.bat
    echo   train_detector.bat train
    echo   train_detector.bat test
    echo.
    echo Optional custom ZIP path:
    echo   train_detector.bat train "D:\path\detector.zip"
    exit /b 2
  )
)

set "OUTPUT=%CD%\runs\detector\%RUN_NAME%"
set "LAST=%OUTPUT%\weights\last.pt"
set "INCOMPLETE=%OUTPUT%\.training_incomplete"
set "CONFIG_MARKER=%OUTPUT%\.training_config.json"
set "RESUME_ARG="

if not exist "%DATASET%" (
  echo.
  echo [ERROR] Detector dataset not found:
  echo   %DATASET%
  echo.
  echo Put your detector ZIP here:
  echo   %CD%\data\dataset\detector_class\ObjectDetector_dataset.zip
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

if exist "%INCOMPLETE%" if exist "%LAST%" (
  python training\detector_resume.py check --marker "%CONFIG_MARKER%" --data "%DATASET%" --model "%MODEL%" --imgsz %IMGSZ% --batch %BATCH% --seed %SEED%
  if not errorlevel 1 (
    echo.
    echo === Interrupted detector run found ===
    echo   %LAST%
    echo Resume metadata matches the current dataset and training configuration.
    echo Training will resume automatically.
    set "RESUME_ARG=--resume "%LAST%""
  ) else (
    echo.
    echo [WARN] Interrupted detector run does not match the current dataset/configuration.
    echo It will NOT be resumed. A fresh training run will start.
    if exist "%OUTPUT%" rmdir /s /q "%OUTPUT%"
  )
)

if not defined RESUME_ARG (
  if exist "%LAST%" del /q "%LAST%"
  if exist "%OUTPUT%\weights\best.pt" del /q "%OUTPUT%\weights\best.pt"
  if exist "%INCOMPLETE%" del /q "%INCOMPLETE%"
  if exist "%LAST%" (
    echo [ERROR] Could not remove stale detector checkpoint: %LAST%
    exit /b 1
  )
  if exist "%OUTPUT%\weights\best.pt" (
    echo [ERROR] Could not remove stale detector checkpoint: %OUTPUT%\weights\best.pt
    exit /b 1
  )
)

if not exist "%OUTPUT%" mkdir "%OUTPUT%"
python training\detector_resume.py write --marker "%CONFIG_MARKER%" --data "%DATASET%" --model "%MODEL%" --imgsz %IMGSZ% --batch %BATCH% --seed %SEED%
if errorlevel 1 exit /b 1
> "%INCOMPLETE%" echo Detector training has not completed successfully yet.

echo.
if /I "%MODE%"=="test" (
  echo === YOLO26s detector smoke test: 3 epochs ===
) else (
  echo === YOLO26s detector training: %EPOCHS% epochs ===
)
echo Model       : %MODEL%
echo Dataset ZIP : %DATASET%
echo Image size  : %IMGSZ%
echo Batch       : %BATCH%
echo Patience    : %PATIENCE%
echo Temporary extraction is deleted only after SUCCESS.
echo.

python training\train_detector.py --data "%DATASET%" --model "%MODEL%" --epochs %EPOCHS% --imgsz %IMGSZ% --batch %BATCH% --workers %WORKERS% --device auto --patience %PATIENCE% --seed %SEED% --name "%RUN_NAME%" %EXTRA_ARGS% %RESUME_ARG%
if errorlevel 1 (
  echo.
  echo Detector training stopped or failed.
  echo The extracted detector dataset is being KEPT for retry/resume.
  echo Run train_detector.bat again after fixing the error.
  exit /b 1
)

if exist "%INCOMPLETE%" del /q "%INCOMPLETE%"

echo.
echo === Cleaning extracted detector dataset ===
python -c "import shutil; from training.detector_dataset_utils import DEFAULT_DETECTOR_EXTRACT_DIR; print('Removing:', DEFAULT_DETECTOR_EXTRACT_DIR); shutil.rmtree(DEFAULT_DETECTOR_EXTRACT_DIR, ignore_errors=True)"
if errorlevel 1 echo [WARN] Could not clean the temporary detector dataset automatically.

echo.
if /I "%MODE%"=="test" (
  echo Smoke test completed. No deploy detector was replaced.
) else (
  echo Detector training completed successfully.
  echo Best detector model: %CD%\models\best_detector.pt
)
endlocal
