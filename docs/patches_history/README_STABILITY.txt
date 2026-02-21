ES Paper Trader - Stability Bundle

What this does (stability-first):
1) Adds pt/ctx_builder.py to centralize ctx creation for pt.loop_core
   (does not change runtime unless you wire it in later).
2) Adds tools/PATCH_GUARD_CUTOVER_ENV.ps1 to prevent the v3 cutover block from running
   unless you explicitly set:  $env:PT_ENABLE_CUTOVER="1"

How to apply:
- Copy pt/ctx_builder.py into your repo at: .\pt\ctx_builder.py
- Copy tools/PATCH_GUARD_CUTOVER_ENV.ps1 into: .\tools\PATCH_GUARD_CUTOVER_ENV.ps1
- Run:
     powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\PATCH_GUARD_CUTOVER_ENV.ps1

Then run legacy loop (default):
  Remove-Item Env:\PT_ENABLE_CUTOVER -ErrorAction SilentlyContinue
  $env:PT_ENABLE_CUTOVER="0"
  cmd /c .\tools\START_TRADER.cmd

Later, when ready for controlled cutover, we wire paper_trader.py to call:
  from pt.ctx_builder import build_ctx
  ctx = build_ctx(...)
  pt_run_loop(ctx)
