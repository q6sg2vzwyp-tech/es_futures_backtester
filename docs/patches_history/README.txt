Bundle contents:
- paper_trader.py (patched: removes old PT_CUTOVER_TO_PT_LOOP v3 block; keeps v4 gated cutover)
- pt/*.py modules (as uploaded)

Install:
1) Unzip into C:\Users\owner\Desktop\es_futures_backtester (overwrite).
2) Run: .\.venv\Scripts\python.exe -m py_compile .\paper_trader.py
3) Enable cutover: set PT_ENABLE_CUTOVER=1
4) Start: cmd /c .\tools\START_TRADER.cmd

Notes:
- v4 cutover is controlled by env var PT_ENABLE_CUTOVER.
- Legacy loop remains as fallback when PT_ENABLE_CUTOVER is not enabled.
