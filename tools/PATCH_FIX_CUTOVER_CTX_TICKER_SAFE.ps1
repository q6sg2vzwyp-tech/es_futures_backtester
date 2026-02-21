# PATCH_FIX_CUTOVER_CTX_TICKER_SAFE.ps1
# Fixes NameError: ticker is not defined in PT cutover ctx.
# - Inserts _pt_ticker = ib.reqMktData(con) near the cutover ctx creation
# - Replaces ticker=ticker with ticker=_pt_ticker
# Creates a timestamped backup.

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$root = Split-Path -Parent $PSScriptRoot
$paper = Join-Path $root "paper_trader.py"

if (!(Test-Path $paper)) { throw "paper_trader.py not found at: $paper" }

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$bak = "$paper.bak_fix_cutover_ticker_safe_$ts"
Copy-Item $paper $bak -Force
Write-Host "[BACKUP] $bak" -ForegroundColor Yellow

$src = Get-Content $paper -Raw -Encoding UTF8

# Heuristic: find the cutover block that calls pt_run_loop(...) and builds ctx=dict(...)
# We expect to see a line 'ticker=ticker,' (or 'ticker = ticker' not likely).
if ($src -notmatch "pt_run_loop") {
  throw "No pt_run_loop reference found; nothing to patch."
}

# Replace ticker=ticker with ticker=_pt_ticker in the ctx dict call
$src2 = $src -replace '(?m)^\s*ticker\s*=\s*ticker\s*,\s*$', '            ticker=_pt_ticker,'

# If no change happened, try a looser replace (in case of different indentation / spacing)
if ($src2 -eq $src) {
  $src2 = $src -replace 'ticker\s*=\s*ticker', 'ticker=_pt_ticker'
}

# Now ensure _pt_ticker is defined before ctx is built.
# Insert after the first occurrence of the qualified contract data / after md_warmup / after con set.
# We'll insert immediately before the line that contains 'ctx = dict(' OR 'ctx = {' inside the cutover try block.
$insert = @"
            # PT cutover: ticker handle for market data
            _pt_ticker = ib.reqMktData(con)
"@

# Insert only if _pt_ticker not already present
if ($src2 -notmatch "_pt_ticker\s*=") {
  # Prefer inserting before a 'ctx = dict(' line within the cutover block.
  $pattern = '(?ms)(#\s*={2,}\s*PT_CUTOVER.*?)(\n\s*try:\s*\n)(.*?)(\n\s*ctx\s*=\s*dict\s*\()'
  if ($src2 -match $pattern) {
    $src2 = [regex]::Replace($src2, $pattern, { param($m)
      return $m.Groups[1].Value + $m.Groups[2].Value + $m.Groups[3].Value + $insert + $m.Groups[4].Value
    }, 1)
  } else {
    # Fallback: insert before first 'ctx = dict(' after any pt_run_loop mention
    $pattern2 = '(?ms)(pt_run_loop.*?\n)(.*?)(\n\s*ctx\s*=\s*dict\s*\()'
    if ($src2 -match $pattern2) {
      $src2 = [regex]::Replace($src2, $pattern2, { param($m)
        return $m.Groups[1].Value + $m.Groups[2].Value + $insert + $m.Groups[3].Value
      }, 1)
    } else {
      throw "Could not find a 'ctx = dict(' location to inject _pt_ticker."
    }
  }
}

Set-Content -Path $paper -Value $src2 -Encoding UTF8
Write-Host "[OK] Patched paper_trader.py: ticker handle injected + ctx updated" -ForegroundColor Green

# Compile check
$py = Join-Path $root ".venv\Scripts\python.exe"
if (!(Test-Path $py)) { $py = "python" }
& $py -m py_compile $paper
if ($LASTEXITCODE -ne 0) { throw "py_compile failed with exit code $LASTEXITCODE" }
Write-Host "[OK] Compile: paper_trader.py" -ForegroundColor Green
