@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0\.."

set PYEXE=.\.venv\Scripts\python.exe
if not exist "%PYEXE%" set PYEXE=python

set IBHOST=127.0.0.1
set IBPORT=4002
set CLIENTID=1111

echo [WD] Watchdog started. Close window to stop.

:loop
if exist ".\run\SHUTDOWN.flag" (
  echo [WD] Shutdown flag present. Exiting watchdog.
  exit /b 0
)
echo [START] "%PYEXE%" -u .\paper_trader.py --host %IBHOST% --port %IBPORT% --clientId %CLIENTID%
"%PYEXE%" -u .\paper_trader.py --host %IBHOST% --port %IBPORT% --clientId %CLIENTID%
set RC=%ERRORLEVEL%
if exist ".\run\SHUTDOWN.flag" (
  echo [WD] Shutdown flag present. Exiting watchdog.
  exit /b 0
)
echo [WD] Trader exited (rc=!RC!). Restarting in 3s...
timeout /t 3 >nul
goto :loop
