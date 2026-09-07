@echo off
setlocal
cd /d "%~dp0"

set "VENV_PYTHON=.venv\Scripts\python.exe"
set "VENV_READY="

if exist "%VENV_PYTHON%" (
  "%VENV_PYTHON%" --version >nul 2>nul
  if not errorlevel 1 set "VENV_READY=1"
)

if /I "%~1"=="setup" goto :setup
if /I "%~1"=="clean" (
  if not defined VENV_READY goto :setup
  goto :clean
)
if not defined VENV_READY goto :setup

goto :dispatch

:setup
set "PYTHON_BOOTSTRAP="
where python >nul 2>nul
if not errorlevel 1 set "PYTHON_BOOTSTRAP=python"

if not defined PYTHON_BOOTSTRAP (
  where py >nul 2>nul
  if not errorlevel 1 set "PYTHON_BOOTSTRAP=py -3"
)

if not defined PYTHON_BOOTSTRAP (
  echo Python was not found. Install Python 3.11 or 3.12 first.
  pause
  exit /b 1
)

if not defined VENV_READY (
  if exist "%VENV_PYTHON%" (
    echo The current .venv is broken; repairing it with the detected Python installation...
    %PYTHON_BOOTSTRAP% -m venv --upgrade ".venv"
    if errorlevel 1 (
      echo Could not repair .venv automatically. Rename or delete the .venv folder, then run start.bat setup again.
      goto :error
    )
  ) else (
    echo Creating .venv virtual environment...
    %PYTHON_BOOTSTRAP% -m venv ".venv"
    if errorlevel 1 goto :error
  )
)

echo Installing/updating dependencies...
"%VENV_PYTHON%" -m pip install --upgrade pip
if errorlevel 1 goto :error
"%VENV_PYTHON%" -m pip install -r requirements.txt
if errorlevel 1 goto :error

if /I "%~1"=="setup" (
  echo.
  echo Environment setup completed.
  exit /b 0
)

goto :dispatch

:dispatch
if /I "%~1"=="ngrok" goto :ngrok
if /I "%~1"=="clean" goto :clean
if /I "%~1"=="configure" goto :configure
if /I "%~1"=="dev" goto :dev
if /I "%~1"=="reload" goto :dev
if not "%~1"=="" (
  echo Invalid option: %~1
  echo Usage: start.bat ^| start.bat dev ^| start.bat setup ^| start.bat clean ^| start.bat configure ^| start.bat ngrok
  exit /b 2
)

"%VENV_PYTHON%" launcher.py
exit /b %errorlevel%

:dev
"%VENV_PYTHON%" launcher.py --reload
exit /b %errorlevel%

:clean
if not exist "%VENV_PYTHON%" goto :setup
echo.
echo Cleaning up the previous process on the configured port...
"%VENV_PYTHON%" launcher.py --kill-port
exit /b %errorlevel%

:configure
"%VENV_PYTHON%" launcher.py --configure
set "EXIT_CODE=%errorlevel%"
if not "%EXIT_CODE%"=="0" pause
exit /b %EXIT_CODE%

:ngrok
"%VENV_PYTHON%" -c "import pyngrok" >nul 2>nul
if errorlevel 1 (
  echo Installing pyngrok for ngrok mode...
  "%VENV_PYTHON%" -m pip install "pyngrok>=8.1,<9.0"
  if errorlevel 1 goto :error
)
"%VENV_PYTHON%" launcher.py --ngrok
set "EXIT_CODE=%errorlevel%"
if not "%EXIT_CODE%"=="0" pause
exit /b %EXIT_CODE%

:error
echo.
echo Setup or startup failed. See the error message above.
pause
exit /b 1
