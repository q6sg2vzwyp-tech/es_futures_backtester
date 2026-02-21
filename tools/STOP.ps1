# STOP.ps1 (v2) - self-contained safe stop for ES Paper Trader stack
$ErrorActionPreference = "SilentlyContinue"

# Resolve project root
$ROOT = if ($PSScriptRoot) { (Resolve-Path (Join-Path $PSScriptRoot "..")).Path } else { (Resolve-Path ".").Path }
Set-Location $ROOT

$RUN = Join-Path $ROOT "run"
New-Item -ItemType Directory -Path $RUN -Force | Out-Null

# 1) Signal shutdown
$flag = Join-Path $RUN "SHUTDOWN.flag"
New-Item -ItemType File -Path $flag -Force | Out-Null
Write-Host "[STOP] Wrote $flag"

# 2) Give the bot time to exit cleanly
Start-Sleep -Seconds 2

# 3) Kill any remaining bot processes (only those tied to this repo / filenames)
$procs = Get-CimInstance Win32_Process |
  Where-Object {
    $_.Name -in @("python.exe","pythonw.exe") -and
    ($_.CommandLine -match "es_futures_backtester" -or $_.CommandLine -match "paper_trader\.py" -or $_.CommandLine -match "watchdog_single\.py")
  }

if ($procs) {
  Write-Host ("[STOP] Killing {0} remaining process(es)..." -f $procs.Count)
  foreach ($p in $procs) {
    Write-Host ("  PID {0} PPID {1} :: {2}" -f $p.ProcessId, $p.ParentProcessId, $p.CommandLine)
    Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
  }
} else {
  Write-Host "[STOP] No bot python processes detected."
}

# 4) Optional cleanup: remove stale lock files (do NOT remove SHUTDOWN.flag)
Remove-Item (Join-Path $RUN "paper_trader.lock") -Force -ErrorAction SilentlyContinue
Remove-Item (Join-Path $RUN "watchdog_single.lock") -Force -ErrorAction SilentlyContinue

Write-Host "[STOP] Done."
