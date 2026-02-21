# PATCH_FIX_CUTOVER_CTX_BARS_SAFE_v2.ps1
# - Restores compilation by avoiding insertion inside any try/except blocks
# - Adds BarBuffer import (if missing)
# - Adds a module-level buffer _PT_BARS_BUF (if missing)
# - Switches cutover ctx bars=... to use _PT_BARS_BUF

$ErrorActionPreference = "Stop"

function Backup-File($path, $tag) {
  $ts = Get-Date -Format yyyyMMdd_HHmmss
  $bak = "$path.bak_$tag`_$ts"
  Copy-Item $path $bak -Force
  Write-Host "[BACKUP] $bak" -ForegroundColor DarkGray
  return $bak
}

$root = Split-Path -Parent $PSScriptRoot
$paper = Join-Path $root "paper_trader.py"
$py = Join-Path $root ".venv\Scripts\python.exe"

if (!(Test-Path $paper)) { throw "paper_trader.py not found at: $paper" }
if (!(Test-Path $py)) { throw "python.exe not found at: $py" }

$src = Get-Content $paper -Raw -Encoding UTF8
Backup-File $paper "fix_cutover_bars_v2" | Out-Null

# 1) Ensure BarBuffer import exists (placed near pt_run_loop import if present)
if ($src -notmatch '(?m)^\s*from\s+strategy_core\s+import\s+BarBuffer\s*$') {
  if ($src -match '(?m)^\s*from\s+pt\.loop_core\s+import\s+run_loop\s+as\s+pt_run_loop\s*$') {
    $src = [regex]::Replace(
      $src,
      '(?m)^(?<line>\s*from\s+pt\.loop_core\s+import\s+run_loop\s+as\s+pt_run_loop\s*)$',
      '${line}`r`nfrom strategy_core import BarBuffer',
      1
    )
    Write-Host "[OK] Added import: strategy_core BarBuffer (after pt_run_loop import)" -ForegroundColor Green
  } else {
    # Fallback: insert after the pt.decision_pipeline import if present
    if ($src -match '(?m)^\s*from\s+pt\.decision_pipeline\s+import\s+decide_and_maybe_place_entry\s*$') {
      $src = [regex]::Replace(
        $src,
        '(?m)^(?<line>\s*from\s+pt\.decision_pipeline\s+import\s+decide_and_maybe_place_entry\s*)$',
        '${line}`r`nfrom strategy_core import BarBuffer',
        1
      )
      Write-Host "[OK] Added import: strategy_core BarBuffer (after pt.decision_pipeline import)" -ForegroundColor Green
    } else {
      # Last resort: add near top after future import
      $src = [regex]::Replace(
        $src,
        '(?m)^(from\s+__future__\s+import\s+annotations\s*)$',
        '$1`r`nfrom strategy_core import BarBuffer',
        1
      )
      Write-Host "[OK] Added import: strategy_core BarBuffer (after __future__ import)" -ForegroundColor Yellow
    }
  }
}

# 2) Ensure module-level buffer exists (safe, no try/except interference)
if ($src -notmatch '(?m)^\s*_PT_BARS_BUF\s*=\s*BarBuffer\(') {
  # insert immediately after the BarBuffer import we just ensured
  $src = [regex]::Replace(
    $src,
    '(?m)^(from\s+strategy_core\s+import\s+BarBuffer\s*)$',
    '$1`r`n`r`n# --- PT CUTOVER: module-level bars buffer for pt.loop_core ---`r`n_PT_BARS_BUF = BarBuffer(maxlen=2048)`r`n# --- END PT CUTOVER bars buffer ---',
    1
  )
  Write-Host "[OK] Added _PT_BARS_BUF = BarBuffer(maxlen=2048) at module scope" -ForegroundColor Green
}

# 3) Patch ONLY the cutover ctx to use _PT_BARS_BUF.
# We'll look for a ctx dict that is used with pt_run_loop(ctx). We'll patch within a local window.
if ($src -match 'pt_run_loop\(') {
  # Find the first occurrence of 'pt_run_loop(' and patch the nearest preceding ctx assignment block.
  $idx = $src.IndexOf("pt_run_loop(")
  $start = [Math]::Max(0, $idx - 4000)
  $end = [Math]::Min($src.Length, $idx + 2000)
  $chunk = $src.Substring($start, $end - $start)

  $chunk2 = $chunk

  # Common forms: bars=bars, or bars=somevar,
  # We only rewrite if the key is literally 'bars=' inside ctx construction.
  $chunk2 = [regex]::Replace($chunk2, '(?m)^\s*bars\s*=\s*[A-Za-z_][A-Za-z0-9_]*\s*,\s*$', '        bars=_PT_BARS_BUF,')
  $chunk2 = [regex]::Replace($chunk2, '(?m)^\s*bars\s*=\s*[A-Za-z_][A-Za-z0-9_]*\s*\)\s*$', '        bars=_PT_BARS_BUF)')

  if ($chunk2 -ne $chunk) {
    $src = $src.Substring(0, $start) + $chunk2 + $src.Substring($end)
    Write-Host "[OK] Patched cutover ctx: bars=_PT_BARS_BUF" -ForegroundColor Green
  } else {
    Write-Host "[WARN] Did not find a ctx bars=... line near pt_run_loop(ctx). No changes made to ctx." -ForegroundColor Yellow
  }
} else {
  Write-Host "[WARN] No pt_run_loop(...) found; not patching ctx bars." -ForegroundColor Yellow
}

# Write back
Set-Content -Path $paper -Value $src -Encoding UTF8

# Compile check
& $py -m py_compile $paper
if ($LASTEXITCODE -ne 0) { throw "py_compile failed with exit code $LASTEXITCODE" }
Write-Host "[OK] Compile: paper_trader.py" -ForegroundColor Green
