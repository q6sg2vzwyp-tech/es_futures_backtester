param()

$ErrorActionPreference = "Stop"
$paper = ".\paper_trader.py"
if (-not (Test-Path $paper)) { throw "Missing $paper" }

# Backup
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$bak = "$paper.bak_disable_v3_$stamp"
Copy-Item $paper $bak -Force
Write-Host "[BACKUP] $bak" -ForegroundColor Yellow

$src = Get-Content $paper -Raw -Encoding UTF8

# Remove/disable the stale v3 cutover block (it causes scope/indent issues and steals execution from v4)
# Capture indent so we preserve nesting.
$pat = '(?ms)^(?<indent>\s*)#\s*=+\s*PT_CUTOVER_TO_PT_LOOP v3\s*=+.*?^(?<indent2>\s*)#\s*=+\s*END PT_CUTOVER_TO_PT_LOOP v3\s*=+\s*$'
if ($src -notmatch $pat) {
  Write-Host "[WARN] No v3 cutover block found (nothing changed)." -ForegroundColor Yellow
} else {
  $rep = '${indent}# ================== PT_CUTOVER_TO_PT_LOOP v3 DISABLED ==================' + "`r`n" +
         '${indent}# Removed to avoid executing stale/indented cutover.' + "`r`n" +
         '${indent}# Use PT_CUTOVER_TO_PT_LOOP v4 below (gated by PT_ENABLE_CUTOVER).' + "`r`n" +
         '${indent}# ================== END PT_CUTOVER_TO_PT_LOOP v3 DISABLED =================='
  $src2 = [regex]::Replace($src, $pat, $rep)
  Set-Content -Path $paper -Value $src2 -Encoding UTF8
  Write-Host "[OK] Disabled v3 cutover block" -ForegroundColor Green
}

# Compile check
& ".\.venv\Scripts\python.exe" -m py_compile $paper
Write-Host "[OK] Compile: $paper" -ForegroundColor Green
