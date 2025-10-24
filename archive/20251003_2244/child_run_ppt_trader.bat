@echo off
setlocal EnableExtensions EnableDelayedExpansion
title PPT_TRADER_WIN
cd /d "C:\Users\owner\Desktop\es_futures_backtester"
:RUN
echo ============================================================
echo [Child] start %date% %time%
echo Running: "C:\Users\owner\AppData\Local\Programs\Python\Python313\python.exe" -X faulthandler -u "paper_trader.py" --host 127.0.0.1 --port 7497 --clientId 19 --conId 637533641 --tp-ladder ""1.0R:50,2.0R:50"" --no-confirm --place-orders --debug-regime
"C:\Users\owner\AppData\Local\Programs\Python\Python313\python.exe" -X faulthandler -u "paper_trader.py" --host 127.0.0.1 --port 7497 --clientId 19 --conId 637533641 --tp-ladder ""1.0R:50,2.0R:50"" --no-confirm --place-orders --debug-regime
set "EXITCODE=%ERRORLEVEL%"
echo [Child] exit code=%EXITCODE%
echo (This window stays open. Press any key to relaunch, or close it)...
pause >nul
goto RUN
