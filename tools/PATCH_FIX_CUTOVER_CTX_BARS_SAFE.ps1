param(
  [string]$ProjectRoot = (Get-Location).Path
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$paper = Join-Path $ProjectRoot "paper_trader.py"
if (!(Test-Path $paper)) { throw "paper_trader.py not found at: $paper" }

$ts = Get-Date -Format yyyyMMdd_HHmmss
Copy-Item $paper "$paper.bak_fix_cutover_bars_safe_$ts" -Force
Write-Host "[BACKUP] $paper.bak_fix_cutover_bars_safe_$ts" -ForegroundColor Yellow

$src = Get-Content $paper -Raw -Encoding UTF8

# Ensure BarBuffer import exists (strategy_core is at repo root, not pt/)
if ($src -notmatch '(?m)^\s*from\s+strategy_core\s+import\s+BarBuffer\b') {
  # Prefer to insert after pt.loop_core import if present, else after other imports near top.
  $ins = "`nfrom strategy_core import BarBuffer`n"
  if ($src -match '(?m)^\s*from\s+pt\.loop_core\s+import\s+run_loop\s+as\s+pt_run_loop\s*$') {
    $src = [regex]::Replace($src, '(?m)^\s*from\s+pt\.loop_core\s+import\s+run_loop\s+as\s+pt_run_loop\s*$', '$0' + $ins, 1)
    Write-Host "[OK] Added import: strategy_core BarBuffer (after pt_run_loop import)" -ForegroundColor Green
  } else {
    # Fallback: insert after typing/ib_insync imports block
    if ($src -match '(?m)^\s*from\s+ib_insync\s+import\s+.*$') {
      $src = [regex]::Replace($src, '(?m)^\s*from\s+ib_insync\s+import\s+.*$', '$0' + $ins, 1)
      Write-Host "[OK] Added import: strategy_core BarBuffer (after ib_insync import)" -ForegroundColor Green
    } else {
      # Last resort: prepend after __future__
      $src = [regex]::Replace($src, '(?m)^(from\s+__future__\s+import\s+[^\r\n]+[\r\n]+)', '$1' + $ins, 1)
      Write-Host "[OK] Added import: strategy_core BarBuffer (after __future__)" -ForegroundColor Green
    }
  }
}

# Patch cutover ctx: create bars and pass it
# We look for the dict(ctx) construction inside the PT cutover block that currently passes bars=bars,
if ($src -notmatch 'bars\s*=\s*bars') {
  Write-Host "[WARN] Could not find 'bars=bars' in paper_trader.py; trying to locate PT cutover ctx block anyway." -ForegroundColor Yellow
}

# Insert _pt_bars creation right before ctx = { ... } or ctx = dict( ... ) inside cutover block.
# We'll locate the first occurrence of 'ctx = {' after 'pt_run_loop' usage marker, and inject before it.
$idx = $src.IndexOf("pt_run_loop")
if ($idx -lt 0) { throw "Could not find pt_run_loop in paper_trader.py (cutover import missing?)" }

$tail = $src.Substring($idx)
# Find ctx assignment
$m = [regex]::Match($tail, '(?ms)^\s*ctx\s*=\s*(dict\s*\(|\{)')
if (!$m.Success) { throw "Could not find ctx assignment after pt_run_loop region." }

$insertPos = $idx + $m.Index
# Determine indentation from matched line
$lineStart = $src.LastIndexOf("`n", $insertPos)
if ($lineStart -lt 0) { $lineStart = 0 } else { $lineStart += 1 }
$linePrefix = $src.Substring($lineStart, $insertPos - $lineStart)
$indent = ""
if ($linePrefix -match '^(?<i>\s*)ctx\s*=') { $indent = $Matches['i'] }

$inject = $indent + "# --- CUTOVER: bars buffer for pt.loop_core ---`n" +
          $indent + "_pt_bars = BarBuffer(maxlen=2048)`n" +
          $indent + "# --- END CUTOVER: bars buffer ---`n"

# Only inject if not already present
if ($src -notmatch '(?m)^\s*_pt_bars\s*=\s*BarBuffer\(') {
  $src = $src.Insert($insertPos, $inject)
  Write-Host "[OK] Injected _pt_bars = BarBuffer(...) before ctx" -ForegroundColor Green
} else {
  Write-Host "[OK] _pt_bars already present; skipping inject" -ForegroundColor Green
}

# Replace bars=bars with bars=_pt_bars in the ctx block (only first occurrence after pt_run_loop region)
$src2 = $src
$src2 = [regex]::Replace($src2, '(?m)\bbars\s*=\s*bars\b', 'bars=_pt_bars', 1)

if ($src2 -ne $src) {
  $src = $src2
  Write-Host "[OK] Patched ctx: bars=_pt_bars" -ForegroundColor Green
} else {
  Write-Host "[WARN] Did not replace bars=bars (maybe already fixed or different formatting)." -ForegroundColor Yellow
}

Set-Content -Path $paper -Value $src -Encoding UTF8

# Compile check
$py = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (!(Test-Path $py)) { $py = "python" }
& $py -m py_compile $paper
if ($LASTEXITCODE -ne 0) { throw "py_compile failed with exit code $LASTEXITCODE" }

Write-Host "[OK] Compile: paper_trader.py" -ForegroundColor Green
