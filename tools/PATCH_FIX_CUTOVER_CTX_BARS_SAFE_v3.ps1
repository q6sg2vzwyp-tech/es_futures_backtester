param()

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$pt = Join-Path $root "paper_trader.py"

if (!(Test-Path $pt)) { throw "paper_trader.py not found at: $pt" }

$ts = Get-Date -Format yyyyMMdd_HHmmss
$bak = "$pt.bak_fix_cutover_bars_v3_$ts"
Copy-Item $pt $bak -Force
Write-Host "[BACKUP] $bak" -ForegroundColor Yellow

$src = Get-Content $pt -Raw -Encoding UTF8

# 1) Ensure BarBuffer import exists (as its own clean line)
if ($src -notmatch '(?m)^\s*from\s+strategy_core\s+import\s+BarBuffer\s*$') {
  if ($src -match '(?m)^\s*from\s+pt\.loop_core\s+import\s+run_loop\s+as\s+pt_run_loop\s*$') {
    $src = [regex]::Replace(
      $src,
      '(?m)^\s*from\s+pt\.loop_core\s+import\s+run_loop\s+as\s+pt_run_loop\s*$',
      '$0' + "`r`n" + 'from strategy_core import BarBuffer',
      1
    )
    Write-Host "[OK] Added import: strategy_core BarBuffer (after pt_run_loop import)" -ForegroundColor Green
  } else {
    # fallback: insert near other imports (after ib_insync import if present)
    $src = [regex]::Replace(
      $src,
      '(?m)^\s*from\s+ib_insync\s+import\s+.*$',
      '$0' + "`r`n" + 'from strategy_core import BarBuffer',
      1
    )
    Write-Host "[OK] Added import: strategy_core BarBuffer (fallback placement)" -ForegroundColor Green
  }
} else {
  Write-Host "[INFO] BarBuffer import already present" -ForegroundColor DarkGray
}

# 2) Ensure a module-scope shared bars buffer exists
if ($src -notmatch '(?m)^\s*_PT_BARS_BUF\s*=\s*BarBuffer\(') {
  if ($src -match '(?m)^\s*from\s+strategy_core\s+import\s+BarBuffer\s*$') {
    $insert = @"
# --- PT CUTOVER shared bars buffer (module-scope) ---
_PT_BARS_BUF = BarBuffer(maxlen=2048)
# --- END PT CUTOVER shared bars buffer ---
"@
    $src = [regex]::Replace(
      $src,
      '(?m)^\s*from\s+strategy_core\s+import\s+BarBuffer\s*$',
      '$0' + "`r`n`r`n" + $insert.TrimEnd(),
      1
    )
    Write-Host "[OK] Added _PT_BARS_BUF at module scope" -ForegroundColor Green
  } else {
    throw "Could not find BarBuffer import line to anchor _PT_BARS_BUF insertion."
  }
} else {
  Write-Host "[INFO] _PT_BARS_BUF already present" -ForegroundColor DarkGray
}

# 3) Patch cutover ctx to use module-scope buffer
# Replace the specific bug pattern where ctx passes bars=bars (undefined).
$before = $src
$src = [regex]::Replace($src, '(?m)(\bbars\s*=\s*)bars(\s*,)', '${1}_PT_BARS_BUF$2')
if ($src -ne $before) {
  Write-Host "[OK] Patched cutover ctx: bars=_PT_BARS_BUF" -ForegroundColor Green
} else {
  Write-Host "[WARN] Did not find pattern 'bars=bars,' to replace. If you still get NameError(bars), paste the cutover ctx block." -ForegroundColor Yellow
}

# Write back
Set-Content -Path $pt -Value $src -Encoding UTF8

# Compile check
$py = Join-Path $root ".venv\Scripts\python.exe"
if (!(Test-Path $py)) { $py = "python" }
& $py -m py_compile $pt
if ($LASTEXITCODE -ne 0) { throw "py_compile failed with exit code $LASTEXITCODE" }

Write-Host "[OK] Compile: paper_trader.py" -ForegroundColor Green
