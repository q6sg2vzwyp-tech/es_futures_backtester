@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

set TS=20260217_192740
set BAKDIR=_bak\%TS%
set PATCHDIR=patches

if not exist "%BAKDIR%" mkdir "%BAKDIR%" >nul 2>nul
if not exist "%PATCHDIR%" mkdir "%PATCHDIR%" >nul 2>nul

echo [CLEAN] Moving backup noise (*.bak_hdr_*) into %BAKDIR% ...
for %%F in (*.bak_hdr_*) do (
  move /y "%%F" "%BAKDIR%\" >nul
  echo   moved %%F
)

echo [CLEAN] Moving older patch scripts (PATCH_*.ps1) into %PATCHDIR% ...
for %%F in (PATCH_*.ps1) do (
  move /y "%%F" "%PATCHDIR%\" >nul
  echo   moved %%F
)

echo [DONE] Tools cleaned.
echo        Backups: tools\%BAKDIR%
echo        Patches: tools\%PATCHDIR%
exit /b 0
