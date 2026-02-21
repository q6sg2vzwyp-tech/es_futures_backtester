param(
  [string]$RepoRoot = (Get-Location).Path
)
$ErrorActionPreference = "Stop"
Set-Location $RepoRoot
$paper = Join-Path $RepoRoot "paper_trader.py"

# Prefer v3 backup, else any cutover backup
$bak = Get-ChildItem "$paper.bak_cutover_pt_runloop_v3_*" -File -ErrorAction SilentlyContinue |
  Sort-Object LastWriteTime -Descending | Select-Object -First 1

if (-not $bak) {
  $bak = Get-ChildItem "$paper.bak_cutover_pt_runloop_*" -File -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
}

if (-not $bak) { throw "No cutover backup found matching: $paper.bak_cutover_pt_runloop_*" }

Copy-Item $bak.FullName $paper -Force
Write-Host "[OK] Restored paper_trader.py from $($bak.Name)" -ForegroundColor Green

$py = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (!(Test-Path $py)) { $py = "python" }
& $py -m py_compile $paper
if ($LASTEXITCODE -ne 0) { throw "py_compile failed with exit code $LASTEXITCODE" }
Write-Host "[OK] Compile: paper_trader.py" -ForegroundColor Green
