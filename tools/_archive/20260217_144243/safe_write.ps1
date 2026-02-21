param(
  [Parameter(Mandatory=$true)][string]$TargetPath,
  [Parameter(Mandatory=$true)][string]$TempPath,
  [int]$MinBytes = 2000
)

Set-StrictMode -Version Latest
$ErrorActionPreference="Stop"

if (!(Test-Path $TempPath)) { throw "Temp file missing: $TempPath" }

$len = (Get-Item $TempPath).Length
if ($len -lt $MinBytes) { throw "Refusing write: temp too small ($len bytes) < $MinBytes" }

# also refuse to overwrite with 0 bytes even if MinBytes lowered accidentally
if ($len -le 0) { throw "Refusing write: temp is empty" }

Move-Item -Force $TempPath $TargetPath
Write-Host "SAFE_WRITE OK -> $TargetPath ($len bytes)"
