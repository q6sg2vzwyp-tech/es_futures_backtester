@echo off
setlocal ENABLEDELAYEDEXPANSION

:: --- Move to this BAT's folder ---
cd /d "%~dp0"

:: --- Resolve Python (prefer .venv) ---
set "VENV_PY=.venv\Scripts\python.exe"
set "PYTHON_EXE="
if exist "%VENV_PY%" set "PYTHON_EXE=%VENV_PY%"
if not defined PYTHON_EXE (
  for %%P in (python.exe py.exe) do (
    where %%P >nul 2>nul && (set "PYTHON_EXE=%%P" & goto :foundpy)
  )
  echo [ERROR] Could not find Python or .venv. Create a venv first.
  pause
  exit /b 1
)
:foundpy

:: --- Check watchdog_single.py exists ---
if not exist "watchdog_single.py" (
  echo [ERROR] watchdog_single.py not found next to this BAT.
  pause
  exit /b 1
)

:: --- Target script and args ---
set "TARGET=paper_trader.py"

:: Default args: paper-only, Thompson, balanced risk, place orders + IB PnL
set "ARGS=--place-orders --use-ib-pnl --bandit thompson --learn-mode advisory --risk-profile balanced"

echo [INFO] Launching watchdog_single.py with %PYTHON_EXE%
"%PYTHON_EXE%" "watchdog_single.py" --python "%PYTHON_EXE%" --target "%cd%\%TARGET%" -- %ARGS%
echo.
echo [INFO] Python exited. Check watchdog.log / watchdog.stderr.log
pause
exit /b 0
