Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ROOT = (Resolve-Path ".").Path
$PY = Join-Path $ROOT ".venv\Scripts\python.exe"
if (!(Test-Path $PY)) { throw "Python venv not found at $PY" }

& $PY .\repair_paper_trader_diag.py --file .\paper_trader.py
