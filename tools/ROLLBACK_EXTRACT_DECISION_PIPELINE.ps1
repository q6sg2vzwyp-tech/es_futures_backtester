param(
  [string]$RepoRoot = (Get-Location).Path
)
$ErrorActionPreference = "Stop"
Set-Location $RepoRoot
$paper = Join-Path $RepoRoot "paper_trader.py"
$bak = Get-ChildItem "$paper.bak_decision_extract_*" -File | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $bak) { throw "No backup found matching: $paper.bak_decision_extract_*" }
Copy-Item $bak.FullName $paper -Force
Write-Host "[OK] Restored paper_trader.py from $($bak.Name)" -ForegroundColor Green
