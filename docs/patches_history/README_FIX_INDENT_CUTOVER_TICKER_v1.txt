ES Paper Trader — Fix IndentationError in cutover ticker snippet (v1)

Symptom:
IndentationError: expected an indented block after 'try' (paper_trader.py around the PT cutover ticker snippet)

What this does:
- Rewrites the block starting at:
    # -- CUTOVER: ticker --
  up to (but not including) the line:
    ctx = dict(
  using clean, correct indentation.
- Creates a backup:
    paper_trader.py.bak_fix_indent_cutover_ticker_<timestamp>
- Runs py_compile on paper_trader.py

Usage (repo root):
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\PATCH_FIX_INDENT_CUTOVER_TICKER.ps1
