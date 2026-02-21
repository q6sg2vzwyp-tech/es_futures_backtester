<# 
RESTORE_PAPER_TRADER_LAST_COMPILES.ps1

Goal:
- You currently have a broken paper_trader.py (IndentationError).
- This script searches paper_trader.py backups in project root (paper_trader.py.bak_*)
  newest-first, and restores the first backup that successfully compiles.

Usage (from project root):
  powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\RESTORE_PAPER_TRADER_LAST_COMPILES.ps1

Notes:
- Uses your project venv if present: .\.venv\Scripts\python.exe
- Creates a safety backup of the current (broken) paper_trader.py before overwriting.
#>

param(
  [string]$ProjectRoot = (Resolve-Path ".").Path
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function _pyexe {
  $venv = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
  if (Test-Path $venv) { return $venv }
  return "python"
}

$paper = Join-Path $ProjectRoot "paper_trader.py"
if (!(Test-Path $paper)) { throw "paper_trader.py not found at: $paper" }

$backups = Get-ChildItem -Path $ProjectRoot -File -Filter "paper_trader.py.bak_*" |
  Sort-Object LastWriteTime -Descending

if (!$backups -or $backups.Count -eq 0) {
  throw "No backups found matching paper_trader.py.bak_* in: $ProjectRoot"
}

$py = _pyexe
Write-Host "[INFO] Using python: $py" -ForegroundColor Cyan
Write-Host "[INFO] Found $($backups.Count) backups. Searching newest-first for a compiling candidate..." -ForegroundColor Cyan

$tmp = Join-Path $ProjectRoot "run\__paper_trader_compile_test__.py"
New-Item -ItemType Directory -Force (Split-Path $tmp -Parent) | Out-Null

$winner = $null
foreach ($b in $backups) {
  try {
    Copy-Item -Force $b.FullName $tmp
    & $py -m py_compile $tmp 2>$null
    if ($LASTEXITCODE -eq 0) {
      $winner = $b
      break
    }
  } catch {
    # ignore and continue
  }
}

Remove-Item $tmp -Force -ErrorAction SilentlyContinue
Remove-Item "$tmp`c" -Force -ErrorAction SilentlyContinue  # pyc sidecar for some envs

if (!$winner) {
  throw "No backup compiled successfully. If you want, run: & $py -m py_compile <backupfile> to inspect."
}

$ts = Get-Date -Format yyyyMMdd_HHmmss
$cur_bak = Join-Path $ProjectRoot ("paper_trader.py.bak_broken_before_restore_" + $ts)
Copy-Item -Force $paper $cur_bak

Copy-Item -Force $winner.FullName $paper

Write-Host "[OK] Restored paper_trader.py from:" -ForegroundColor Green
Write-Host "     $($winner.FullName)" -ForegroundColor Green
Write-Host "[BACKUP] Saved previous (broken) version to:" -ForegroundColor Yellow
Write-Host "     $cur_bak" -ForegroundColor Yellow

# Final compile check on real file
& $py -m py_compile $paper
if ($LASTEXITCODE -ne 0) { throw "Restored file did not compile (unexpected). Check: $paper" }

Write-Host "[OK] Compile: paper_trader.py" -ForegroundColor Green
