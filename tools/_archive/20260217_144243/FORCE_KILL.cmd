\
@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0\.."

echo =========================================
echo   FORCE_KILL - paper_trader.py
echo =========================================

REM Clear shutdown flag
if exist ".\run\SHUTDOWN.flag" (
  del /f /q ".\run\SHUTDOWN.flag" >nul 2>nul
  echo [OK] Cleared .\run\SHUTDOWN.flag
)

echo [INFO] Killing python.exe/pythonw.exe where CommandLine contains paper_trader.py ...

for /f "tokens=2 delims=," %%P in ('
  wmic process where "name='python.exe' and CommandLine like '%%paper_trader.py%%'" get ProcessId /format:csv ^| findstr /r /c:",[0-9]"
') do (
  echo   taskkill /PID %%P /F
  taskkill /PID %%P /F >nul 2>nul
)

for /f "tokens=2 delims=," %%P in ('
  wmic process where "name='pythonw.exe' and CommandLine like '%%paper_trader.py%%'" get ProcessId /format:csv ^| findstr /r /c:",[0-9]"
') do (
  echo   taskkill /PID %%P /F
  taskkill /PID %%P /F >nul 2>nul
)

echo.
echo [CHECK] Remaining matches (expect NONE for python.exe/pythonw.exe):
wmic process where "CommandLine like '%%paper_trader.py%%'" get Name,ProcessId,CommandLine 2>nul

echo =========================================
echo DONE
echo =========================================
exit /b 0
