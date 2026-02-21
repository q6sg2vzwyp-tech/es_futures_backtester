ES Paper Trader — Rewrite PT cutover block (v1)

Fixes:
- IndentationError around the PT cutover block (common after repeated regex patches).
- Makes cutover block defensive: pulls optional ctx keys if they exist to avoid NameError churn.

Usage (repo root):
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\PATCH_REWRITE_PT_CUTOVER_BLOCK.ps1

Notes:
- Creates backup: paper_trader.py.bak_rewrite_cutover_<timestamp>
- Replaces the entire cutover block up to the legacy "while True:" line.
- If cutover throws, it logs [pt_cutover_err] and falls through to legacy loop (so you can still run).
