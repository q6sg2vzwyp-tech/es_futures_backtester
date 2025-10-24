@echo off
setlocal enableextensions
cd /d "%~dp0"

set "PY_EXE=C:\Users\owner\AppData\Local\Programs\Python\Python313\python.exe"
if not exist "%PY_EXE%" set "PY_EXE=py"

if not exist "logs" mkdir "logs"

"%PY_EXE%" "%cd%\watchdog_single.py" --tee-console --log-dir "%cd%\logs" --restart-delay 3 --max-restarts 1 --cwd "%cd%" -- ^
  "%PY_EXE%" -u -c "import sys, time; print('watchdog OK'); sys.stdout.flush(); time.sleep(1)"

echo.
echo ===== If you see "watchdog OK" above, your watchdog works. =====
pause
endlocal
