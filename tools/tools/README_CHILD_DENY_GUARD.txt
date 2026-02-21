PT Child-Deny Guard (Safe Patch v1)

Purpose
- Prevents paper_trader.py from running as a child process of another paper_trader.py.
- Stops the parent->child self-spawn chain so you don't end up with two traders.

How it works
- Inserts a tiny Python guard near the top of paper_trader.py.
- Uses psutil (if available) to inspect the parent process cmdline.
- If parent cmdline contains 'paper_trader.py', exits immediately.
- Override: set environment variable PT_ALLOW_CHILD=1.

Install
1) Unzip into your project root:
   C:\Users\owner\Desktop\es_futures_backtester

2) Run patch:
   tools\PATCH_PT_CHILD_DENY_GUARD_SAFE_v1.ps1
   (or tools\RUN_PATCH_CHILD_DENY_GUARD.cmd)

Safety
- Refuses to patch if paper_trader.py is <100000 bytes.
- Writes a backup to backups\paper_trader.py_YYYYMMDD_HHMMSS.bak
- Idempotent: won't insert twice.

After patch
- Compile:
  .\.venv\Scripts\python.exe -m py_compile .\paper_trader.py
- Run and confirm only one paper_trader exists:
  Get-CimInstance Win32_Process | ? { $_.CommandLine -match "paper_trader\.py" } | select ProcessId,ParentProcessId,CommandLine
