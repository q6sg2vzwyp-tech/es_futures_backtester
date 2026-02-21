param(
  [string]$RepoRoot = (Get-Location).Path
)
$ErrorActionPreference = "Stop"
Set-Location $RepoRoot

$paper = Join-Path $RepoRoot "paper_trader.py"
if (!(Test-Path $paper)) { throw "Missing: $paper" }

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
Copy-Item $paper "$paper.bak_fix_indent_cutover_ticker_$ts" -Force
Write-Host "[BACKUP] $paper.bak_fix_indent_cutover_ticker_$ts" -ForegroundColor DarkGray

$src = Get-Content $paper -Raw -Encoding UTF8

# Locate marker indentation
$m = [regex]::Match($src, '(?m)^(?<ind>\s*)#\s*--\s*CUTOVER:\s*ticker\s*--\s*$')
if (-not $m.Success) { throw "Could not find marker: # -- CUTOVER: ticker --" }
$ind = $m.Groups["ind"].Value
$ind2 = $ind + "    "

# Replace region from marker line through the line just before 'ctx = dict('
$pattern = '(?ms)^(?<ind>\s*)#\s*--\s*CUTOVER:\s*ticker\s*--\s*$.*?^(?<ctxind>\s*)ctx\s*=\s*dict\s*\(\s*$'

# Build replacement with explicit concatenation (avoid accidental $var: parsing)
$nl = "`r`n"
$replacement =
  ($ind + "# -- CUTOVER: ticker --" + $nl) +
  ($ind + "# loop_core expects a live ib_insync Ticker object. Create it here." + $nl) +
  ($ind + "try:" + $nl) +
  ($ind2 + "_pt_ticker = ib.reqMktData(con)" + $nl) +
  ($ind2 + "try:" + $nl) +
  ($ind2 + "    ib.sleep(0.2)" + $nl) +
  ($ind2 + "except Exception:" + $nl) +
  ($ind2 + "    pass" + $nl) +
  ($ind + "except Exception:" + $nl) +
  ($ind2 + "_pt_ticker = None" + $nl + $nl) +
  ($ind + "ctx = dict(" + $nl)

$src2 = [regex]::Replace($src, $pattern, $replacement, 1)
if ($src2 -eq $src) { throw "Failed to rewrite ticker snippet region (pattern not matched as expected)." }

Set-Content -Path $paper -Value $src2 -Encoding UTF8

$py = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (!(Test-Path $py)) { $py = "python" }
& $py -m py_compile $paper
if ($LASTEXITCODE -ne 0) { throw "py_compile failed with exit code $LASTEXITCODE" }

Write-Host "[OK] Fixed indentation around CUTOVER ticker snippet + compile OK" -ForegroundColor Green
