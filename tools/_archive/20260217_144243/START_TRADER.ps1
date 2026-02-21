param(
  [string]$IbHost = "127.0.0.1",
  [int]$Port = 4002,
  [int]$ClientId = 1111,
  [switch]$PlaceOrders
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# clear shutdown flag
$flag = Join-Path $root "run\SHUTDOWN.flag"
Remove-Item $flag -Force -ErrorAction SilentlyContinue

$py = Join-Path $root ".venv\Scripts\python.exe"
if (!(Test-Path $py)) { throw "Missing venv python: $py" }

$args = @("-u", ".\paper_trader.py", "--host", $IbHost, "--port", "$Port", "--clientId", "$ClientId")
if ($PlaceOrders) { $args += "--place-orders" }

Write-Host "[START] $py $($args -join ' ')" -ForegroundColor Green

& $py @args
exit $LASTEXITCODE
