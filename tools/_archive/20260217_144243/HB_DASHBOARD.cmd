@echo off
setlocal EnableExtensions

cd /d "%~dp0.."

echo -----------------------------------------
echo   ES Heartbeat Dashboard (Monitor Only)
echo -----------------------------------------

".\.venv\Scripts\python.exe" ".\hb_monitor.py" --alt-screen --interval 1.0 --no-singleton

echo.
echo Dashboard exited.
pause
endlocal
