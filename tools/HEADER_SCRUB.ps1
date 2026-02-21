param(
  [string]$ToolsDir = (Join-Path (Get-Location).Path "tools")
)
$ErrorActionPreference = "Stop"
if (!(Test-Path $ToolsDir)) { throw "Missing tools dir: $ToolsDir" }

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$bakDir = Join-Path $ToolsDir "_bak_hdr_$ts"
New-Item -ItemType Directory -Force $bakDir | Out-Null

$files = Get-ChildItem $ToolsDir -File -Include *.ps1,*.cmd,*.bat
$fixed = 0
foreach ($f in $files) {
  $lines = Get-Content $f.FullName -Encoding UTF8
  if ($lines.Count -gt 0 -and $lines[0].Trim() -eq '\') {
    Copy-Item $f.FullName (Join-Path $bakDir $f.Name) -Force
    if ($lines.Count -gt 1) {
      Set-Content -Path $f.FullName -Value ($lines[1..($lines.Count-1)]) -Encoding UTF8
    } else {
      Set-Content -Path $f.FullName -Value @() -Encoding UTF8
    }
    Write-Host "[FIX] $($f.Name)" -ForegroundColor Green
    $fixed++
  }
}
Write-Host "[DONE] Header scrub complete. Fixed=$fixed Backups=$bakDir" -ForegroundColor Cyan
