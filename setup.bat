@echo off
setlocal enabledelayedexpansion

:: Look for the first match of mu_venv-* in the expected directory
set FOUND=
for /D %%f in ("%LocalAppData%\python\mu\mu_venv-*") do (
    set FOUND=%%f
    goto :found
)

echo Mu virtual environment not found.
pause
exit /b 1

:found
echo Found Mu environment: !FOUND!

set ACTIVATE_PATH=!FOUND!\Scripts\activate.bat
set DEACTIVATE_PATH=!FOUND!\Scripts\deactivate.bat
set PIP_PATH=!FOUND!\Scripts\pip.exe
set PYTHON_PATH=!FOUND!\Scripts\python.exe

echo Using Python from: !PYTHON_PATH!

:: Upgrade pip using Mu's Python
echo Upgrading pip...
"!PYTHON_PATH!" -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo Failed to upgrade pip.
    pause
    exit /b 1
)

:: Install specific versions of dependencies
echo Installing dependencies...
"!PIP_PATH!" install tensorflow numpy sounddevice
if %errorlevel% neq 0 (
    echo Failed to install dependencies.
    pause
    exit /b 1
)

echo Setup complete and all dependencies installed correctly!
pause
exit /b 0
