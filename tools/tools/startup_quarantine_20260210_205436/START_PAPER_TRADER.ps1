Set-StrictMode -Version Latest
$ErrorActionPreference="Stop"
$ROOT = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ROOT
& "$ROOT\.venv\Scripts\python.exe" -u "$ROOT\paper_trader.py" --host 127.0.0.1 --port 4002 --clientId 1111
