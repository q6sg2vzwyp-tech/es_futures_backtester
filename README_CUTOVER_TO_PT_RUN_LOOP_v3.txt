ES Paper Trader — Cutover to pt.loop_core.run_loop v3

This fixes v2's mistake where the import might be inserted inside a try/except block.
v3 inserts the import ONLY after a top-level anchor import line:
- from pt.decision_pipeline import decide_and_maybe_place_entry
(fallback: from pt.ai_hooks import AIHooks)

Apply:
  powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\PATCH_CUTOVER_TO_PT_RUN_LOOP_v3.ps1

Rollback:
  powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\ROLLBACK_CUTOVER_TO_PT_RUN_LOOP_v3.ps1
