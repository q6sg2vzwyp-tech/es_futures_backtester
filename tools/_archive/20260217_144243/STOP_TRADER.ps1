param(
  [switch]$HardKill
)

$ErrorActionPreference = "Continue"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# signal graceful shutdown
$runDir = Join-Path $root "run"
New-Item -ItemType Directory -Force -Path $runDir | Out-Null
$flag = Join-Path $runDir "SHUTDOWN.flag"
"shutdown $(Get-Date -Format o)" | Out-File -FilePath $flag -Encoding utf8 -Force
Write-Host "[STOP] Wrote $flag" -ForegroundColor Yellow

if ($HardKill) {
  Write-Host "[STOP] Hard-killing remaining bot python process(es)..." -ForegroundColor Red
  Get-CimInstance Win32_Process |
    Where-Object {
      $_.Name -in @("python.exe","pythonw.exe") -and
      $_.CommandLine -like "*es_futures_backtester*" -and
      $_.CommandLine -match "paper_trader\.py"
    } |
    ForEach-Object {
      Write-Host ("  PID {0} :: {1}" -f $_.ProcessId, $_.CommandLine) -ForegroundColor Red
      try { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue } catch {}
    }
}

Write-Host "[STOP] Done." -ForegroundColor Yellow

