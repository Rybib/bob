@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"

set "APP_NAME=BOB"
set "APP_VERSION=project-builder-v5-kokoro"
set "LOG_DIR=logs"
set "LOG_FILE=%LOG_DIR%\launcher-windows.log"
set "VENV_DIR=.venv"
set "PY=%VENV_DIR%\Scripts\python.exe"
set "MARKER=%VENV_DIR%\.bob_deps_installed"
set "PY_BOOT="
set "STEP=0"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%" >nul 2>&1
if not exist "models" mkdir "models" >nul 2>&1
if not exist "projects" mkdir "projects" >nul 2>&1
break > "%LOG_FILE%"

call :header
call :step "Checking Python"
call :ensure_python || exit /b 1
echo.

call :step "Preparing Bob"
call :install_dependencies || exit /b 1
echo.

call :step "Checking models and setup"
echo     If this is the first launch, BOB will open the setup wizard.
echo     The wizard downloads the selected LLM, Kokoro, Whisper, and OpenWakeWord.
echo.

call :step "Starting BOB"
echo     When setup is complete, the assistant opens in this window.
echo.
"%PY%" bob.py
if errorlevel 1 (
  call :fail "BOB exited with an error."
  exit /b 1
)

echo.
echo BOB closed.
call :pause_close
exit /b 0

:header
cls
echo ============================================================
echo                         Launch BOB
echo               Local voice assistant installer
echo ============================================================
echo.
echo Version: %APP_VERSION%
echo Folder:  %CD%
echo Log:     %CD%\%LOG_FILE%
echo.
exit /b 0

:step
set /a STEP+=1
echo [%STEP%] %~1
exit /b 0

:pause_close
echo.
pause
exit /b 0

:fail
echo.
echo BOB could not finish this step:
echo   %~1
echo.
echo The detailed log is here:
echo   %CD%\%LOG_FILE%
echo.
echo After fixing the issue, double-click Launch Bob Windows.bat again.
call :pause_close
exit /b 1

:run_quiet
echo + %* >> "%LOG_FILE%"
%* >> "%LOG_FILE%" 2>&1
exit /b %errorlevel%

:detect_python
set "PY_BOOT="
py -3 --version >nul 2>&1
if not errorlevel 1 (
  set "PY_BOOT=py -3"
  exit /b 0
)

python --version >nul 2>&1
if not errorlevel 1 (
  set "PY_BOOT=python"
  exit /b 0
)

exit /b 1

:ensure_python
call :detect_python
if not errorlevel 1 (
  for /f "tokens=*" %%v in ('%PY_BOOT% --version 2^>^&1') do echo     Found %%v
  exit /b 0
)

echo     Python 3 is missing.
echo     BOB can try to install Python automatically with winget.
echo     This may open a Windows installer prompt.
echo.
choice /C YN /N /M "Install Python now? [Y/N] "
if errorlevel 2 (
  call :fail "Python 3 is required. Install Python 3.10 or newer, then run this launcher again."
  exit /b 1
)

where winget >nul 2>&1
if errorlevel 1 (
  call :fail "winget was not found. Install Python 3.10 or newer from python.org, then run this launcher again."
  exit /b 1
)

echo     Installing Python with winget...
echo + winget install Python.Python.3.12 >> "%LOG_FILE%"
winget install --id Python.Python.3.12 -e --source winget --accept-package-agreements --accept-source-agreements >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
  call :fail "Python installation failed."
  exit /b 1
)

set "PATH=%LocalAppData%\Programs\Python\Python312;%LocalAppData%\Programs\Python\Python312\Scripts;%PATH%"
call :detect_python
if errorlevel 1 (
  call :fail "Python installed, but it was not found on PATH. Close this window and run the launcher again."
  exit /b 1
)

for /f "tokens=*" %%v in ('%PY_BOOT% --version 2^>^&1') do echo     Found %%v
exit /b 0

:install_dependencies
if not exist "%PY%" (
  echo     Creating local Python environment...
  call :run_quiet %PY_BOOT% -m venv "%VENV_DIR%"
  if errorlevel 1 (
    call :fail "Could not create the local Python environment."
    exit /b 1
  )
) else (
  echo     Local Python environment already exists.
)

set "NEEDS_INSTALL=0"
if not exist "%MARKER%" set "NEEDS_INSTALL=1"

if "%NEEDS_INSTALL%"=="1" (
  echo     Installing/updating packages. This can take several minutes.
  echo     Detailed install output is hidden in logs\launcher-windows.log.

  call :run_quiet "%PY%" -m pip install --upgrade pip wheel setuptools
  if errorlevel 1 (
    call :fail "Could not upgrade pip."
    exit /b 1
  )

  echo     Installing llama-cpp-python...
  call :run_quiet "%PY%" -m pip install llama-cpp-python --prefer-binary --no-cache-dir
  if errorlevel 1 (
    call :fail "Could not install llama-cpp-python. You may need Microsoft C++ Build Tools, then run this launcher again."
    exit /b 1
  )

  call :run_quiet "%PY%" -m pip install -r requirements.txt
  if errorlevel 1 (
    call :fail "Could not install BOB's Python packages."
    exit /b 1
  )

  break > "%MARKER%"
  echo     Dependencies are ready.
) else (
  echo     Dependencies are already ready.
)
exit /b 0
