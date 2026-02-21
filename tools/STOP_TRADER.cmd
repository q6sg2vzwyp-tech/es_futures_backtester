@echo off
setlocal
cd /d "%~dp0\.."

REM Write shutdown flag (graceful)
if not exist ".\run" mkdir ".\run" >nul 2>nul
echo stop> ".\run\SHUTDOWN.flag"
echo [STOP] Wrote .\run\SHUTDOWN.flag
echo [STOP] Done.
exit /b 0
