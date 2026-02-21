FIX_DIAG_CONNECT_SYNTAX.zip

Purpose:
- Repairs the invalid one-line try/except inserted into paper_trader.py by DIAG_CONNECT_TARGET patch.
- Converts it into a proper multiline try/except block so Python can parse it.

Usage:
1) Extract into your project root:
   C:\Users\owner\Desktop\es_futures_backtester

2) Run in PowerShell:
   cd C:\Users\owner\Desktop\es_futures_backtester
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
   .\FIX_DIAG_CONNECT_SYNTAX.ps1

3) Compile check:
   .\.venv\Scripts\python.exe -c "import py_compile; py_compile.compile('paper_trader.py', doraise=True); print('paper_trader OK')"

Rollback:
- The script saves a backup in:
  tools\patches_quarantine_YYYYMMDD_HHMMSS\paper_trader.py.BEFORE_FIX_DIAG_SYNTAX
