ES Paper Trader — Fix IndentationError in cutover ticker snippet (v2)

This version fixes a PowerShell parsing bug in v1.

Usage (repo root):
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\PATCH_FIX_INDENT_CUTOVER_TICKER.ps1

What it changes:
- Only the block starting at:
    # -- CUTOVER: ticker --
  up to the line:
    ctx = dict(
  is rewritten with correct indentation.
- Creates backup:
    paper_trader.py.bak_fix_indent_cutover_ticker_<timestamp>
- Runs py_compile.
