@echo on
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"

set "SCRIPT_DIR=%~dp0"
set "VENV_PY=%SCRIPT_DIR%.venv\Scripts\python.exe"
set "PY=python"
if exist "%VENV_PY%" set "PY=%VENV_PY%"

echo ==== PATH CHECKS ====
if not exist "%PY%" (
  echo [ERROR] Python not found: "%PY%"
  pause & exit /b 1
)

if not exist "%SCRIPT_DIR%watchdog_single.py" (
  echo [ERROR] Missing: %SCRIPT_DIR%watchdog_single.py
  dir "%SCRIPT_DIR%"
  pause & exit /b 1
)

set "ARGS_FILE=%SCRIPT_DIR%args.txt"
set "ARGS="
if exist "%ARGS_FILE%" for /f usebackq tokens=* %%L in ("%ARGS_FILE%") do set "ARGS=%%L"

set "LOGDIR=%SCRIPT_DIR%logs"
if not exist "%LOGDIR%" mkdir "%LOGDIR%"
for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "STAMP=%%I"
set "LOG=%LOGDIR%\bot_stdout_%STAMP%.log"

echo ==== LAUNCH PLAN ====
echo PY: %PY%
echo ARGS(file): %ARGS%
echo ARGS(extra): %*
echo LOG: %LOG%
echo ======================

"%PY%" -u "%SCRIPT_DIR%watchdog_single.py" %ARGS% %*
echo ==== EXIT RC=%ERRORLEVEL% ====
pause
endlocal & exit /b %ERRORLEVEL%
