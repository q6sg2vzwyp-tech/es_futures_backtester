@echo off
echo =========================================
echo   ES PAPER TRADER - CLEAN START
echo =========================================
echo.

call "%~dp0FORCE_KILL.cmd"

echo Starting trader...
".\.venv\Scripts\python.exe" -u .\paper_trader.py --host 127.0.0.1 --port 4002 --clientId 1111

echo.
echo Trader exited.
