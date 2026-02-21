Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ROOT = Split-Path -Parent $PSScriptRoot
Set-Location $ROOT

try {
  cmd /c ".\tools\HB_DASHBOARD.cmd"
}
catch {
  Write-Host "HB dashboard failed: $($_.Exception.Message)"
  pause
}
