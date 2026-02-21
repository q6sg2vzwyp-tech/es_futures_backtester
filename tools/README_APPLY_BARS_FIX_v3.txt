Apply Bars Fix v3 (safe)

1) From project root:
   powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\PATCH_FIX_CUTOVER_CTX_BARS_SAFE_v3.ps1

2) Then:
   cmd /c .\tools\START_TRADER.cmd

If you get NameError for another ctx field, paste the [pt_cutover_err] line and the traceback.
