@echo off
setlocal ENABLEDELAYEDEXPANSION
REM --- Move to repo root ---
cd /d "%~dp0"

REM --- Resolve Python (prefer venv) ---
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

REM --- Require watchdog_single.py ---
if not exist "watchdog_single.py" (
  echo [ERROR] watchdog_single.py not found next to this BAT.
  pause
  exit /b 1
)

REM --- Bot target ---
set "TARGET=paper_trader.py"

REM --- Runtime args (NO --success-exits anywhere) ---
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
  --max-contracts 1 ^
  --risk-pct 0.01 ^
  --risk-ticks 12 ^
  --tp-R 1.0 ^
  --tif GTC ^
  --outsideRth ^
  --preflight-whatif ^
  --margin-buffer-pct 0.10 ^
  --orphan-sweep-delay-sec 8 ^
  --max-trades-per-day 999 ^
  --no-day-guard ^
  --no-loss-cap ^
  --peak-dd-guard-pct 0"

echo [INFO] Launching watchdog_single.py with %PYTHON_EXE%
REM --- Log + keep window open ---
set "LOGDIR=logs"
if not exist "%LOGDIR%" mkdir "%LOGDIR%"
for /f "tokens=1-4 delims=/ " %%a in ("%date%") do set "D=%%d%%b%%c"
for /f "tokens=1-2 delims=:." %%a in ("%time%") do set "T=%%a%%b"
set "LOG=%LOGDIR%\watchdog_%D%_%T%.log"
echo [INFO] Logging to %LOG%
powershell -NoProfile -Command ^
  "$p = Start-Process -FilePath '%PYTHON_EXE%' -ArgumentList @('watchdog_single.py','--python','%PYTHON_EXE%','--bot','%cd%\%TARGET%','--',%ARGS%) -NoNewWindow -PassThru; " ^
  "Wait-Process -InputObject $p; " ^
  "Write-Host '--- process ended ---'; " ^
  "Read-Host 'Press Enter to close'"
pause
endlocal
