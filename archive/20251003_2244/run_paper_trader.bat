@echo off
REM ================ PAPER (TWS Paper / IB Gateway Paper) =================
REM This connects to PAPER TWS/Gateway (port 7497). Orders are LIVE but in paper.

setlocal enabledelayedexpansion
cd /d "%~dp0"

REM ---- Connection (Paper uses 7497 by default) ----
set HOST=127.0.0.1
set PORT=7497
set CLIENTID=9

REM ---- Strategy / Risk / Execution ----
set QTY=1
set FAST=8
set SLOW=20
set RSI=14

REM Smart anti-chop + risk (from upgraded bot)
set ADX_MIN=20
set BB_BW_MIN=0.06
set ATR_PCT_MIN=0.012
set STOP_ENTRY_ATR=0.5
set ATR_INIT_MULT=1.8
set CHAND_MULT=3
set CHAND_LEN=22
set TIME_STOP_BARS=30
set PROGRESS_NEEDED_R=0.5
set PARTIAL_FRAC=0.5
set R_TAKE_PARTIAL=1.0
set COOLDOWN_SIGNALS=3
REM Disable thin hours (Chicago time). Empty to allow all.
set DISABLE_HOURS=0-3

REM Gates / trading hours
set MIN_ATR_TICKS=0
set MAX_SPREAD_TICKS=4
set RTH_ONLY=--rth-only
set FLAT_AT=14:57

REM Market data type: use delayed if you don’t have live ES data
set USE_DELAYED=--use-delayed

REM Verbose console
set VERBOSE=--verbose

echo.
echo ================== PAPER MODE ==================
echo Host:%HOST%  Port:%PORT%  ClientId:%CLIENTID%
echo Paper account required (TWS/Gateway set to Paper). Orders WILL be sent (paper).
echo Type YES to continue, or anything else to abort.
set /p OK=Confirm (YES/no): 
if /I not "%OK%"=="YES" goto :abort

REM --- Run (place-orders is required for live paper orders) ---
py paper_trader.py ^
  --host %HOST% --port %PORT% --clientId %CLIENTID% ^
  --place-orders %USE_DELAYED% %RTH_ONLY% ^
  --qty %QTY% --fast %FAST% --slow %SLOW% --rsi %RSI% ^
  --min-atr-ticks %MIN_ATR_TICKS% --max-spread-ticks %MAX_SPREAD_TICKS% ^
  --flat-at %FLAT_AT% ^
  --adx-min %ADX_MIN% --bb-bw-min %BB_BW_MIN% --atr-pct-min %ATR_PCT_MIN% ^
  --stop-entry-atr-mult %STOP_ENTRY_ATR% --atr-init-mult %ATR_INIT_MULT% ^
  --chand-mult %CHAND_MULT% --chand-len %CHAND_LEN% ^
  --time-stop-bars %TIME_STOP_BARS% --progress-needed-r %PROGRESS_NEEDED_R% ^
  --partial-frac %PARTIAL_FRAC% --r-take-partial %R_TAKE_PARTIAL% ^
  --cooldown-signals %COOLDOWN_SIGNALS% ^
  --disable-hours %DISABLE_HOURS% ^
  %VERBOSE%

echo.
echo Paper trader exited. Press any key to close.
pause >nul
endlocal
goto :eof

:abort
echo Aborted. Press any key to close.
pause >nul
endlocal
