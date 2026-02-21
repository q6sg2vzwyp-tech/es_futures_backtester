@echo off
echo =========================================
echo   ES PAPER TRADER - FORCE KILL
echo =========================================
echo.

echo Killing ALL python.exe instances running paper_trader.py...
for /f "tokens=2 delims=," %%P in ('
  wmic process where "name='python.exe' and commandline like '%%paper_trader.py%%'" get processid /format:csv ^| findstr /r /c:",[0-9]"
') do (
  echo Killing PID %%P
  taskkill /PID %%P /F >nul 2>nul
)

for /f "tokens=2 delims=," %%P in ('
  wmic process where "name='pythonw.exe' and commandline like '%%paper_trader.py%%'" get processid /format:csv ^| findstr /r /c:",[0-9]"
') do (
  echo Killing PID %%P
  taskkill /PID %%P /F >nul 2>nul
)

echo.
echo Removing shutdown flag...
if exist ".\run\SHUTDOWN.flag" del ".\run\SHUTDOWN.flag"

echo.
echo DONE. Mutex should now be released.
echo =========================================
