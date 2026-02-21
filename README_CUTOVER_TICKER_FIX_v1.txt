ES Paper Trader — Cutover Hotfix (ticker) v1

Fixes:
- NameError: ticker is not defined in PT cutover ctx
  Adds _pt_ticker = ib.reqMktData(con) and passes ticker=_pt_ticker into ctx.

Also includes:
- tools\HEADER_SCRUB.ps1 : robust PowerShell header scrubber (removes a stray leading "\" line).

Usage (repo root):
1) Apply ticker fix:
   powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\PATCH_FIX_CUTOVER_CTX_TICKER.ps1

2) (Optional) scrub tool headers:
   powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\HEADER_SCRUB.ps1
