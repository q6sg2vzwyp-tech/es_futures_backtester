Fix pack: DecisionPipeline import newline issue

Symptom:
  SyntaxError showing: from pt.ai_hooks import AIHooks`r`nfrom pt.decision_pipeline ...

Cause:
  The patch inserted a literal PowerShell escape sequence (`r`n) into Python.

Fix:
  Run (from repo root):
    powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\FIX_IMPORT_NEWLINE.ps1

This will:
  - backup paper_trader.py
  - replace literal `r`n sequences with real newlines
  - run py_compile and fail if it still errors
