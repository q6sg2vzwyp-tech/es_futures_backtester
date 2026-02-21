param(
  [string]$RepoRoot = (Get-Location).Path
)
$ErrorActionPreference = "Stop"
Set-Location $RepoRoot

$paper = Join-Path $RepoRoot "paper_trader.py"
if (!(Test-Path $paper)) { throw "Missing: $paper" }

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
Copy-Item $paper "$paper.bak_fix_cutover_logger_$ts" -Force
Write-Host "[BACKUP] $paper.bak_fix_cutover_logger_$ts" -ForegroundColor DarkGray

$src = Get-Content $paper -Raw -Encoding UTF8

# Replace ctx logger=logger with logger=log inside the PT cutover block only.
# This avoids touching any other "logger=" occurrences.
$pattern = '(?s)(# ================== PT_CUTOVER_TO_PT_LOOP v3 ==================.*?ctx = dict\(\s*)(.*?)(\s*\)\s*\r?\n\s*return\s+int\(pt_run_loop\(ctx\)\))'
$m = [regex]::Match($src, $pattern)
if (-not $m.Success) { throw "Could not locate PT_CUTOVER_TO_PT_LOOP v3 block." }

$block = $m.Value
if ($block -match '(?m)^\s*logger\s*=\s*log\s*,\s*$') {
  Write-Host "[OK] Cutover ctx already uses logger=log" -ForegroundColor Green
} else {
  $block2 = [regex]::Replace($block, '(?m)^\s*logger\s*=\s*logger\s*,\s*$', '        logger=log,', 1)
  if ($block2 -eq $block) {
    # fallback: if indentation differs, do a softer replace
    $block2 = $block -replace 'logger=logger,', 'logger=log,'
  }
  if ($block2 -notmatch 'logger\s*=\s*log') { throw "Failed to patch logger in cutover ctx." }
  $src = $src.Replace($block, $block2)
  Write-Host "[OK] Patched cutover ctx: logger=log" -ForegroundColor Green
}

Set-Content -Path $paper -Value $src -Encoding UTF8

$py = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (!(Test-Path $py)) { $py = "python" }
& $py -m py_compile $paper
if ($LASTEXITCODE -ne 0) { throw "py_compile failed with exit code $LASTEXITCODE" }
Write-Host "[OK] Compile: paper_trader.py" -ForegroundColor Green
