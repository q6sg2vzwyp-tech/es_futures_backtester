@echo off
setlocal ENABLEDELAYEDEXPANSION

:: Move to repo
cd /d "%~dp0"

:: Resolve Python
set "VENV_PY=.venv\Scripts\python.exe"
set "PYTHON_EXE="
if exist "%VENV_PY%" set "PYTHON_EXE=%VENV_PY%"
if not defined PYTHON_EXE (
  for %%P in (python.exe py.exe) do (
    where %%P >nul 2>nul && (set "PYTHON_EXE=%%P" & goto :foundpy)
  )
  echo [ERROR] Python not found (.venv or system).
  pause
  exit /b 1
)
:foundpy

if not exist "watchdog_single.py" (
  echo [ERROR] watchdog_single.py not found.
  pause
  exit /b 1
)

set "TARGET=paper_trader.py"

:: Supported "uncap" set: disable day guard; set peak-dd guard pct to 0
set "ARGS=^
  --place-orders ^
  --use-ib-pnl ^
  --local-symbol ESZ5 ^
  --force-delayed ^
  --force-midpoint-rt ^
  --poll-hist-when-no-rt ^
  --rt-starve-sec 2 ^
  --trade-start-ct 00:00 ^
  --trade-end-ct 23:59 ^
  --host 127.0.0.1 --port 7497 --clientId 360 ^
  --risk-profile balanced ^
  --start-contracts 1 ^
  --max-contracts 6 ^
  --risk-pct 0.01 ^
  --risk-ticks 12 ^
  --tp-R 1.0 ^
  --tif GTC ^
  --preflight-whatif ^
  --margin-buffer-pct 0.05 ^
  --orphan-sweep-delay-sec 8 ^
  --max-trades-per-day 30 ^
  --day-loss-cap-R 999 ^
  --no-loss-cap ^
  --no-day-guard ^
  --peak-dd-guard-pct 0"

echo [INFO] Launching watchdog_single.py with %PYTHON_EXE%
"%PYTHON_EXE%" "watchdog_single.py" --python "%PYTHON_EXE%" --bot "%cd%\%TARGET%" -- %ARGS%
echo.
echo [INFO] Python exited.
pause
exit /b 0
