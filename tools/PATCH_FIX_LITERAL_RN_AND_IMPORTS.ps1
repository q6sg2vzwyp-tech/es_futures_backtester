param(
  [string]$ProjectRoot = (Get-Location).Path
)

$ErrorActionPreference = "Stop"

$paper = Join-Path $ProjectRoot "paper_trader.py"
if (-not (Test-Path $paper)) { throw "paper_trader.py not found at $paper" }

# Backup
$ts = Get-Date -Format yyyyMMdd_HHmmss
$bak = "$paper.bak_fix_literalrn_$ts"
Copy-Item $paper $bak -Force
Write-Host "[BACKUP] $bak" -ForegroundColor Yellow

$src = Get-Content $paper -Raw -Encoding UTF8

# 1) Fix literal PowerShell-escaped newline tokens that were inserted into Python:
#    The file currently contains the *literal* characters: `r`n (backtick r backtick n)
#    Replace that sequence with a real Windows CRLF newline.
$literal = [regex]::Escape('`r`n')
if ($src -match $literal) {
  $src = [regex]::Replace($src, $literal, "`r`n")
  Write-Host "[OK] Replaced literal `r`n tokens with real CRLF newlines" -ForegroundColor Green
} else {
  Write-Host "[INFO] No literal `r`n tokens found" -ForegroundColor DarkGray
}

# 2) Hardening: if two imports got concatenated without a newline, split them.
#    Example bad line:
#      from pt.loop_core import run_loop as pt_run_loopfrom strategy_core import BarBuffer
$src = $src -replace '(from\s+pt\.loop_core\s+import\s+run_loop\s+as\s+pt_run_loop)\s*(from\s+strategy_core\s+import\s+BarBuffer)', '$1' + "`r`n" + '$2'

# 3) Optional: ensure BarBuffer import exists once (we don't add it here—only de-mangle).
#    This avoids creating duplicates or changing order too much.

Set-Content -Path $paper -Value $src -Encoding UTF8
Write-Host "[OK] Wrote paper_trader.py" -ForegroundColor Green

# Compile check
$py = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
& $py -m py_compile $paper
if ($LASTEXITCODE -ne 0) { throw "py_compile failed (rc=$LASTEXITCODE)" }
Write-Host "[OK] Compile: paper_trader.py" -ForegroundColor Green
