ES Paper Trader — Cutover to pt.loop_core.run_loop v2

Use this AFTER:
- pt\loop_core.py contains def run_loop(ctx: Dict[str, Any]) -> int:
- Your tools start/stop are stable.

What it does
- Adds: from pt.loop_core import run_loop as pt_run_loop
- Inserts a delegation block immediately above the first legacy "while True:" loop.
- Leaves the legacy loop in place but unreachable (return before it).
- Creates a timestamped backup and runs py_compile (hard-fail on error).

Apply
  powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\PATCH_CUTOVER_TO_PT_RUN_LOOP_v2.ps1

Rollback
  powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\ROLLBACK_CUTOVER_TO_PT_RUN_LOOP_v2.ps1
