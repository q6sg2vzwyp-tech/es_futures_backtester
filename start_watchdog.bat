@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ====== Paths ======
set ROOT=%~dp0
cd /d "%ROOT%"
set LOGDIR=%ROOT%logs
if not exist "%LOGDIR%" mkdir "%LOGDIR%"

REM ====== Python (venv if present) ======
if exist ".venv\Scripts\python.exe" (
  set "PY=.venv\Scripts\python.exe"
) else (
  set "PY=python"
)

REM ====== Common args (edit these) ======
REM You can tweak these in one place. Keep it on one line or use ^ for new lines.
set "ARGS= --local-symbol ESZ5 --place-orders --use-ib-pnl --learn-mode advisory --enable-arms trend,breakout --param-arms "A:risk_ticks=10,tp_R=1.0,entry_slippage_ticks=1;B:risk_ticks=12,tp_R=1.2,entry_slippage_ticks=2;C:risk_ticks=8,tp_R=0.8,entry_slippage_ticks=1" --trade-start-ct 00:00 --trade-end-ct 23:59 --poll-hist-when-no-rt --poll-interval-sec 10 --rt-staleness-sec 45 --session-reset-cts 08:30,16:00,17:00 --vwap-reset-on-session --short-guard-vwap --short-guard-lower-high --pos-age-cap-sec 1200 --pos-age-minR 0.5 --day-loss-cap-R 3 --max-trades-per-day 12 --max-consec-losses 3 --hwm-stepdown --hwm-stepdown-dollars 5000 --news-bulletin-listen --news-kill-minutes 15 --segment-trade-csv ".\logs\trades_segmented.csv""

REM ====== Optional: paper-only safety (default). Remove to allow live (NOT RECOMMENDED).
REM set "ARGS=%ARGS% --allow_live"

REM ====== Restart/backoff settings ======
set MIN_BACKOFF=5
set MAX_BACKOFF=60
set BACKOFF=%MIN_BACKOFF%

:LOOP
for /f "tokens=1-3 delims=/: " %%a in ("%date% %time%") do (
  set DTS=%%c-%%a-%%b_!time:~0,2!!time:~3,2!!time:~6,2!
)
set LOG=%LOGDIR%\run_!DTS!.log

echo.
echo [START] %DATE% %TIME%  -> logging to "%LOG%"
echo ----------------------------------------------------- >> "%LOG%"
echo [ENV] PY=%PY% >> "%LOG%"
echo [ENV] ROOT=%ROOT% >> "%LOG%"
echo [ENV] LOGDIR=%LOGDIR% >> "%LOG%"
echo [CMD] %PY% paper_trader.py %ARGS% >> "%LOG%"
echo ----------------------------------------------------- >> "%LOG%"

REM === Launch trader and tee console into log ===
"%PY%" paper_trader.py %ARGS% 1>>"%LOG%" 2>&1

set EXITCODE=%ERRORLEVEL%
echo [EXIT] code=%EXITCODE% at %DATE% %TIME% >> "%LOG%"

REM === Self-backups happen inside the script, but make a dated copy as extra belt & suspenders ===
if exist ".\backups" (
  for /f "tokens=1-3 delims=/: " %%a in ("%date% %time%") do (
    set BKTS=%%c%%a%%b_!time:~0,2!!time:~3,2!!time:~6,2!
  )
  copy /y ".\paper_trader.py" ".\backups\paper_trader_!BKTS!.py" >nul 2>&1
)

REM === Backoff & restart ===
if %EXITCODE% EQU 0 (
  echo [INFO] Clean exit; restarting in %MIN_BACKOFF%s...
  set BACKOFF=%MIN_BACKOFF%
) else (
  echo [WARN] Crash/err exit %EXITCODE%; backoff %BACKOFF%s...
  REM Exponential-ish backoff
  if %BACKOFF% LSS %MAX_BACKOFF% (
    set /a BACKOFF=BACKOFF*2
    if %BACKOFF% GTR %MAX_BACKOFF% set BACKOFF=%MAX_BACKOFF%
  )
)

REM Sleep BACKOFF seconds
powershell -NoProfile -Command "Start-Sleep -Seconds %BACKOFF%"
goto :LOOP
