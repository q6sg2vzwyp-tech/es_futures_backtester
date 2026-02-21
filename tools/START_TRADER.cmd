@echo off
setlocal
cd /d "%~dp0\.."

REM Clear shutdown flag if present
if exist ".\run\SHUTDOWN.flag" del /f /q ".\run\SHUTDOWN.flag" >nul 2>nul

set PYEXE=.\.venv\Scripts\python.exe
if not exist "%PYEXE%" set PYEXE=python

set IBHOST=127.0.0.1
set IBPORT=4002
set CLIENTID=1111

echo [START] "%PYEXE%" -u .\paper_trader.py --host %IBHOST% --port %IBPORT% --clientId %CLIENTID%
"%PYEXE%" -u .\paper_trader.py --host %IBHOST% --port %IBPORT% --clientId %CLIENTID%
exit /b %ERRORLEVEL%
