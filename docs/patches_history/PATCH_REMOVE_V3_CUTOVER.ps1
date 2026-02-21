Param(
  [string]$Root = "."
)

$ErrorActionPreference = "Stop"
$paper = Join-Path $Root "paper_trader.py"
if (!(Test-Path $paper)) { throw "Not found: $paper" }

# Backup
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$bak = "$paper.bak_remove_v3_$ts"
Copy-Item $paper $bak -Force
Write-Host "[BACKUP] $bak" -ForegroundColor Yellow

$src = Get-Content $paper -Raw -Encoding UTF8

# Remove the entire legacy PT_CUTOVER v3 block (it is indented incorrectly and fights v4)
$pattern = "(?ms)^\s*# ================== PT_CUTOVER_TO_PT_LOOP v3 ==================.*?^\s*# ================== END PT_CUTOVER_TO_PT_LOOP v3 ==================\s*\r?\n"
$dst = [regex]::Replace($src, $pattern, "`r`n")

if ($dst -eq $src) {
  Write-Host "[WARN] No v3 cutover block found (nothing removed)." -ForegroundColor Yellow
} else {
  Set-Content -Path $paper -Value $dst -Encoding UTF8
  Write-Host "[OK] Removed PT_CUTOVER_TO_PT_LOOP v3 block" -ForegroundColor Green
}

# Compile check
& (Join-Path $Root ".venv\Scripts\python.exe") -m py_compile $paper
Write-Host "[OK] Compile: $paper" -ForegroundColor Green
