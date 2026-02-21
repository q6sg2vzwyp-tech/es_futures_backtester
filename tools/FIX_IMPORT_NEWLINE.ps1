param(
  [string]$RepoRoot = (Get-Location).Path
)

$ErrorActionPreference = "Stop"
Set-Location $RepoRoot

$paper = Join-Path $RepoRoot "paper_trader.py"
if (!(Test-Path $paper)) { throw "Missing: $paper" }

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
Copy-Item $paper "$paper.bak_fix_import_newline_$ts" -Force
Write-Host "[BACKUP] $paper.bak_fix_import_newline_$ts" -ForegroundColor DarkGray

$src = Get-Content $paper -Raw -Encoding UTF8

# Replace literal PowerShell escape sequences accidentally inserted into Python:
#   `r`n  -> actual newline
$src2 = $src -replace '`r`n', "`r`n"

if ($src2 -eq $src) {
  Write-Host "[WARN] No literal ``r``n sequences found. Nothing changed." -ForegroundColor Yellow
} else {
  Set-Content -Path $paper -Value $src2 -Encoding UTF8
  Write-Host "[OK] Replaced literal ``r``n with real newlines in paper_trader.py" -ForegroundColor Green
}

# Compile check (fail hard if python returns nonzero)
$py = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (!(Test-Path $py)) { $py = "python" }

& $py -m py_compile $paper
if ($LASTEXITCODE -ne 0) { throw "py_compile failed with exit code $LASTEXITCODE" }
Write-Host "[OK] Compile: paper_trader.py" -ForegroundColor Green
