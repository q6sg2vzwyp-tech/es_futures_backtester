@echo off
setlocal EnableExtensions EnableDelayedExpansion
title ES Paper Trader — Gate Summary

rem Jump to this script's folder
cd /d "%~dp0"

echo === Summarizing latest IB paper session ===
where python >nul 2>&1 || (
  echo.
  echo [ERROR] Python not found in PATH. Install Python 3 or open this in your Python prompt.
  pause
  exit /b 1
)

rem Run the Python summarizer
python summarize_gates.py
if errorlevel 1 (
  echo.
  echo [ERROR] summarize_gates.py returned an error. See messages above.
  pause
  exit /b 1
)

rem Find newest session folder under results\
set "SESSDIR="
for /f "delims=" %%D in ('dir /b /ad /o-d "results\ib_session_*" 2^>nul') do (
  set "SESSDIR=%%D"
  goto :found
)
:found
if not defined SESSDIR (
  echo.
  echo [ERROR] No session folders found under results\
  pause
  exit /b 1
)

set "SUMMARY=results\%SESSDIR%\gate_summary.txt"
set "REASONS=results\%SESSDIR%\reason_counts.csv"
set "REGSTATS=results\%SESSDIR%\regime_stats.csv"

echo.
echo Latest session: results\%SESSDIR%

if exist "%SUMMARY%" (
  echo Opening gate_summary.txt ...
  start "" "%SUMMARY%"
  rem Also show it in Explorer for quick access to all outputs
  start "" explorer.exe /e,/select,"%SUMMARY%"
) else (
  echo [WARN] Could not find gate_summary.txt
)

if exist "%REASONS%" start "" "%REASONS%"
if exist "%REGSTATS%" start "" "%REGSTATS%"

echo.
echo Done.
pause
