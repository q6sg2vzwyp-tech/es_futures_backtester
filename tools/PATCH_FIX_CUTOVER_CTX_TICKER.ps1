param(
  [string]$RepoRoot = (Get-Location).Path
)
$ErrorActionPreference = "Stop"
Set-Location $RepoRoot

$paper = Join-Path $RepoRoot "paper_trader.py"
if (!(Test-Path $paper)) { throw "Missing: $paper" }

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
Copy-Item $paper "$paper.bak_fix_cutover_ticker_$ts" -Force
Write-Host "[BACKUP] $paper.bak_fix_cutover_ticker_$ts" -ForegroundColor DarkGray

$src = Get-Content $paper -Raw -Encoding UTF8

# Locate PT cutover block and ensure ticker is defined and passed.
$blockPattern = '(?s)(# ================== PT_CUTOVER_TO_PT_LOOP v3 ==================.*?# ================== END PT_CUTOVER_TO_PT_LOOP v3 ==================)'
$m = [regex]::Match($src, $blockPattern)
if (-not $m.Success) { throw "Could not locate PT_CUTOVER_TO_PT_LOOP v3 block." }

$block = $m.Groups[1].Value

# Insert ticker creation before ctx dict if missing
if ($block -notmatch '(?m)^\s*#\s*--\s*CUTOVER:\s*ticker\s*--\s*$') {
  $block2 = [regex]::Replace(
    $block,
    '(?m)^\s*ctx\s*=\s*dict\s*\(\s*$',
@'
        # -- CUTOVER: ticker --
        # loop_core expects a live ib_insync Ticker object. Create it here.
        try:
            _pt_ticker = ib.reqMktData(con)
            try:
                ib.sleep(0.2)
            except Exception:
                pass
        except Exception:
            _pt_ticker = None

        ctx = dict(
'@,
    1
  )
  if ($block2 -eq $block) { throw "Failed to insert ticker creation before ctx dict." }
  $block = $block2
  Write-Host "[OK] Inserted ticker creation (_pt_ticker)" -ForegroundColor Green
} else {
  Write-Host "[OK] Ticker creation already present" -ForegroundColor Green
}

# Patch ctx ticker key to use _pt_ticker
if ($block -match 'ticker\s*=\s*ticker\s*,') {
  $block = [regex]::Replace($block, '(?m)^\s*ticker\s*=\s*ticker\s*,\s*$', '        ticker=_pt_ticker,', 1)
} else {
  $block = $block -replace 'ticker=ticker,', 'ticker=_pt_ticker,'
}

if ($block -notmatch '(?m)^\s*ticker\s*=\s*_pt_ticker\s*,') { throw "Failed to patch ctx ticker=_pt_ticker." }
Write-Host "[OK] Patched ctx: ticker=_pt_ticker" -ForegroundColor Green

# Write back
$src = $src.Substring(0, $m.Index) + $block + $src.Substring($m.Index + $m.Length)
Set-Content -Path $paper -Value $src -Encoding UTF8

$py = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (!(Test-Path $py)) { $py = "python" }
& $py -m py_compile $paper
if ($LASTEXITCODE -ne 0) { throw "py_compile failed with exit code $LASTEXITCODE" }
Write-Host "[OK] Compile: paper_trader.py" -ForegroundColor Green
