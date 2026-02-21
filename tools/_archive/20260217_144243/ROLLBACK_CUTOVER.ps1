param(
  [string]$RepoRoot = (Get-Location).Path
)

$ErrorActionPreference = "Stop"

$target = Join-Path $RepoRoot "paper_trader.py"
if (!(Test-Path $target)) { throw "Cannot find $target. Run from repo root." }

$backs = Get-ChildItem -Path $RepoRoot -Filter "paper_trader.py.bak_cutover_*" -File |
  Sort-Object LastWriteTime -Descending

if (!$backs -or $backs.Count -eq 0) {
  throw "No paper_trader.py.bak_cutover_* backups found. Nothing to rollback."
}

$latest = $backs[0].FullName
Copy-Item $latest $target -Force
Write-Host "[OK] Restored paper_trader.py from: $latest" -ForegroundColor Green

