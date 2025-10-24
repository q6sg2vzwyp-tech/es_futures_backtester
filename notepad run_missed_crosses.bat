@echo off
cd /d "%~dp0"
python -u missed_crosses.py
echo.
echo Done. CSVs saved in the latest session folder under results\.
pause
