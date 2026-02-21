@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo [SCRUB] Removing a stray leading "\" first-line from tools scripts (cmd/bat/ps1) if present...

for %%F in (*.cmd *.bat *.ps1) do (
  for /f "usebackq delims=" %%L in ("%%F") do (
    if "%%L"=="\" (
      set "ts=%DATE:~-4%%DATE:~4,2%%DATE:~7,2%_%TIME:~0,2%%TIME:~3,2%%TIME:~6,2%"
      set "ts=!ts: =0!"
      copy "%%F" "%%F.bak_hdr_!ts!" >nul
      more +1 "%%F" > "%%F.__tmp__"
      move /y "%%F.__tmp__" "%%F" >nul
      echo [FIX] %%F
    )
    goto :nextfile
  )
  :nextfile
)

echo [DONE] Header scrub complete.
exit /b 0
