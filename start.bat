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
  echo Khong tim thay Python. Hay cai Python 3.11 hoac 3.12 truoc.
  pause
  exit /b 1
)

if not defined VENV_READY (
  if exist "%VENV_PYTHON%" (
    echo Moi truong .venv hien tai bi hong, dang sua bang Python vua tim thay...
    %PYTHON_BOOTSTRAP% -m venv --upgrade ".venv"
    if errorlevel 1 (
      echo Khong the tu dong sua .venv. Hay doi ten hoac xoa thu muc .venv roi chay lai start.bat setup.
      goto :error
    )
  ) else (
    echo Dang tao moi truong .venv...
    %PYTHON_BOOTSTRAP% -m venv ".venv"
    if errorlevel 1 goto :error
  )
)

echo Dang cai/cap nhat thu vien...
"%VENV_PYTHON%" -m pip install --upgrade pip
if errorlevel 1 goto :error
"%VENV_PYTHON%" -m pip install -r requirements.txt
if errorlevel 1 goto :error

if /I "%~1"=="setup" (
  echo.
  echo Da cai dat xong moi truong.
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
  echo Tuy chon khong hop le: %~1
  echo Dung: start.bat ^| start.bat dev ^| start.bat setup ^| start.bat clean ^| start.bat configure ^| start.bat ngrok
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
echo Dang don tien trinh cu tren port da cau hinh...
"%VENV_PYTHON%" launcher.py --kill-port
exit /b %errorlevel%

:configure
"%VENV_PYTHON%" launcher.py --configure
set "EXIT_CODE=%errorlevel%"
if not "%EXIT_CODE%"=="0" pause
exit /b %EXIT_CODE%

:ngrok
"%VENV_PYTHON%" launcher.py --ngrok
set "EXIT_CODE=%errorlevel%"
if not "%EXIT_CODE%"=="0" pause
exit /b %EXIT_CODE%

:error
echo.
echo Cai dat hoac khoi dong that bai. Xem thong bao loi phia tren.
pause
exit /b 1
