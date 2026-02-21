@echo off
setlocal
cd /d "%~dp0\.."
powershell -NoProfile -ExecutionPolicy Bypass -File ".\tools\PATCH_IB_CONNECT_RETRIES_SAFE_v2.ps1"
endlocal
