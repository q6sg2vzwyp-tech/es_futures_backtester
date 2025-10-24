@echo off
setlocal
cd /d "%~dp0"

echo Analyzing latest session under "results\"...
python -u analyze_session.py

echo.
echo Done. Press any key to close.
pause >nul
endlocal
