ES Paper Trader — Decision Pipeline Extraction v1

What this does
- Adds pt\decision_pipeline.py (new module)
- Replaces the "if cand:" decision/AI/guardrails/placement block in paper_trader.py
  with a call to decide_and_maybe_place_entry(...)
- Behavior-preserving goal: keep runtime identical while shrinking paper_trader.py

Install
1) Unzip into your repo root: C:\Users\owner\Desktop\es_futures_backtester
   It should place:
     - pt\decision_pipeline.py
     - tools\PATCH_EXTRACT_DECISION_PIPELINE.ps1
     - tools\ROLLBACK_EXTRACT_DECISION_PIPELINE.ps1

Run patch (from repo root)
  powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\PATCH_EXTRACT_DECISION_PIPELINE.ps1

Rollback
  powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\ROLLBACK_EXTRACT_DECISION_PIPELINE.ps1

Notes
- Patch locates the block by the unique marker line:
    "# 1) Bandit choice as before"
- A backup is created:
    paper_trader.py.bak_decision_extract_YYYYMMDD_HHMMSS
