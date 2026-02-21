DIAG_CONNECT_TARGET

What it does:
- Inserts two JSON log entries:
  1) evt=diag_connect_target right before connect_with_retries() is called in main
  2) evt=diag_ib_connect_call right before ib.connect(...) inside connect_with_retries

Why:
- You are seeing [CONNECT] Attempt ... -> clientId=111 even though single.cmdline.txt shows 1111.
- You are also seeing ConnectionRefused even though manual python -c ib.connect(...) succeeds.

This patch will print the *exact* host/port/clientId/timeout values paper_trader is actually using at runtime.

How to use:
1) Copy APPLY_DIAG_CONNECT_TARGET.ps1 into the project root (same folder as paper_trader.py)
2) Run:
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
   .\APPLY_DIAG_CONNECT_TARGET.ps1
3) Run your normal launch:
   .\.venv\Scripts\python.exe .\paper_trader.py @argsTok
4) Paste the first ~30 log lines that include:
   - "evt": "diag_connect_target"
   - "evt": "diag_ib_connect_call"
   - the first "[CONNECT] Attempt ..."

Rollback:
- Restore from tools\patches_quarantine_YYYYMMDD_HHMMSS\paper_trader.py.BEFORE_DIAG
