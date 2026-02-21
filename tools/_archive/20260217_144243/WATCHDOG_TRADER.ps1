param(
  [string]$IbHost = "127.0.0.1",
  [int]$Port = 4002,
  [int]$ClientId = 1111,
  [switch]$PlaceOrders,
  [int]$RestartDelaySec = 3
)

$ErrorActionPreference = "Continue"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$start = Join-Path $PSScriptRoot "START_TRADER.ps1"
if (!(Test-Path $start)) { throw "Missing: $start" }

Write-Host "[WD] Watchdog started. Ctrl+C to stop." -ForegroundColor Cyan

while ($true) {
  $flag = Join-Path $root "run\SHUTDOWN.flag"
  if (Test-Path $flag) {
    Write-Host "[WD] Shutdown flag present. Exiting watchdog." -ForegroundColor Yellow
    break
  }

  Write-Host "[WD] Launching trader..." -ForegroundColor Cyan
  try {
    $argList = @(
      "-NoProfile","-ExecutionPolicy","Bypass",
      "-File", $start,
      "-IbHost", $IbHost,
      "-Port", $Port,
      "-ClientId", $ClientId
    )
    if ($PlaceOrders) { $argList += "-PlaceOrders" }

    $p = Start-Process powershell -ArgumentList $argList -PassThru -WindowStyle Normal
    $p.WaitForExit()
    $code = $p.ExitCode
    Write-Host "[WD] Trader exited (code=$code)." -ForegroundColor Yellow
  } catch {
    Write-Host "[WD] Launch error: $($_.Exception.Message)" -ForegroundColor Red
  }

  Start-Sleep -Seconds $RestartDelaySec
}
