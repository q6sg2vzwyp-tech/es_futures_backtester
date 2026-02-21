@echo off
setlocal
cd /d %~dp0\..
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\PATCH_PT_CHILD_DENY_GUARD_SAFE_v1.ps1
endlocal
