IB Connect Retries Patch (Drop-In) v2

What this fixes:
- Removes stray "from __future__ import annotations" from pt\ib_connect.py (if present).
- Adds a retry-aware helper to pt\ib_connect.py:
    connect_existing_ib_from_args(ib, args, client_id=cid, logger=log)
- Rewrites the connect_with_retries connect line in paper_trader.py to call the helper,
  preserving indentation (so try: blocks remain valid).

How to use:
1) Unzip into ANY folder on your PC.
2) Copy these into your project:
   - tools\PATCH_IB_CONNECT_RETRIES_SAFE_v2.ps1  -> C:\Users\owner\Desktop\es_futures_backtester\tools\
   - tools\RUN_PATCH_IB_CONNECT_V2.cmd           -> C:\Users\owner\Desktop\es_futures_backtester\tools\

3) Run from project root (or double-click the cmd in tools):
   powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\PATCH_IB_CONNECT_RETRIES_SAFE_v2.ps1

4) Verify:
   .\.venv\Scripts\python.exe -m py_compile .\paper_trader.py .\pt\ib_connect.py

Safety:
- Refuses to run if paper_trader.py < 100000 bytes.
- Creates backups in .\backups\
