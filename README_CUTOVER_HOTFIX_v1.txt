ES Paper Trader — Cutover Hotfix (logger) + Tools Header Scrub

1) Fix NameError: logger not defined in PT cutover ctx
   powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\PATCH_FIX_CUTOVER_CTX_LOGGER.ps1

2) If you still see: '\' is not recognized...
   run:
     cmd /c .\tools\HEADER_SCRUB.cmd
   This removes a stray leading "\" line from tools scripts.

3) Updated clean cmd runners included (optional overwrite):
   tools\START_TRADER.cmd
   tools\STOP_TRADER.cmd
   tools\WATCHDOG_TRADER.cmd

Note: In PowerShell, to delete the shutdown flag use:
  Remove-Item .\run\SHUTDOWN.flag -Force -ErrorAction SilentlyContinue
(or use cmd /c del /f /q .\run\SHUTDOWN.flag)
