\
@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

set TS=20260217_193355
set BAKDIR=_bak_hdr\%TS%
if not exist "%BAKDIR%" mkdir "%BAKDIR%" >nul 2>nul

echo [SCRUB] Removing leading "\" first-line from tools scripts (if present)...
for %%F in (*.cmd *.bat *.ps1) do (
  if exist "%%F" (
    for /f "usebackq delims=" %%L in ("%%F") do (
      set FIRST=%%L
      goto :gotfirst_%%F
    )
  )
  :gotfirst_%%F
  if "!FIRST!"=="\" (
    echo   [FIX] %%F
    copy /y "%%F" "%BAKDIR%\%%F" >nul
    more +1 "%%F" > "%%F.__tmp"
    move /y "%%F.__tmp" "%%F" >nul
  )
  set FIRST=
)

echo [DONE] Header scrub complete. Backups in tools\%BAKDIR%
exit /b 0
