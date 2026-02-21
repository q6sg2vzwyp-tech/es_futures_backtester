Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ROOT = (Resolve-Path ".").Path
if (-not (Test-Path (Join-Path $ROOT "pt"))) { throw "pt/ folder not found in $ROOT" }
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$bk = Join-Path $ROOT ("tools\patches_quarantine_" + $ts)
New-Item -ItemType Directory -Force -Path $bk | Out-Null
Copy-Item -Force (Join-Path $ROOT "pt\utils_time.py") (Join-Path $bk "utils_time.py.bak") -ErrorAction SilentlyContinue
Copy-Item -Force (Join-Path $PSScriptRoot "pt\utils_time.py") (Join-Path $ROOT "pt\utils_time.py")
Write-Host "Applied fix. Backup in: $bk"
