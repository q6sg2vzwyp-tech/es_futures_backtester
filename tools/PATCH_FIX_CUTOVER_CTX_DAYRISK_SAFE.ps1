param(
  [string]$Root = "."
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$paper = Join-Path $Root "paper_trader.py"
if (-not (Test-Path $paper)) { throw "paper_trader.py not found at $paper" }

$ts = Get-Date -Format yyyyMMdd_HHmmss
Copy-Item $paper "$paper.bak_fix_cutover_dayrisk_$ts" -Force
Write-Host "[BACKUP] $paper.bak_fix_cutover_dayrisk_$ts" -ForegroundColor Yellow

$src = Get-Content $paper -Raw -Encoding UTF8

# 1) Ensure we have a usable DayRisk object at module scope.
# We insert right after the BarBuffer import if present, else after pt_run_loop import,
# else after the pt.decision_pipeline import block.
$insertBlock = @"
# ---- PT CUTOVER SUPPORT: day_risk object (module-scope) ----
_PT_DAY_RISK = None
try:
    # DayRisk is defined in this module in most builds
    _PT_DAY_RISK = DayRisk()
except Exception:
    try:
        # Fallback: if day_risk already exists for some reason
        _PT_DAY_RISK = globals().get("day_risk", None)
    except Exception:
        _PT_DAY_RISK = None
# ---- END PT CUTOVER SUPPORT: day_risk ----

"@

if ($src -notmatch '(?m)^_PT_DAY_RISK\s*=') {
  $idx = -1

  $m = [regex]::Match($src, '(?m)^(from\s+strategy_core\s+import\s+BarBuffer\s*)$')
  if ($m.Success) { $idx = $m.Index + $m.Length }

  if ($idx -lt 0) {
    $m = [regex]::Match($src, '(?m)^(from\s+pt\.loop_core\s+import\s+run_loop\s+as\s+pt_run_loop\s*)$')
    if ($m.Success) { $idx = $m.Index + $m.Length }
  }

  if ($idx -lt 0) {
    $m = [regex]::Match($src, '(?m)^(from\s+pt\.decision_pipeline\s+import\s+decide_and_maybe_place_entry\s*)$')
    if ($m.Success) { $idx = $m.Index + $m.Length }
  }

  if ($idx -lt 0) { throw "Could not find an anchor import to insert _PT_DAY_RISK block." }

  $src = $src.Insert($idx, "`r`n`r`n" + $insertBlock)
  Write-Host "[OK] Inserted _PT_DAY_RISK module-scope block" -ForegroundColor Green
} else {
  Write-Host "[OK] _PT_DAY_RISK already present; no insert" -ForegroundColor Green
}

# 2) Patch the PT run_loop cutover ctx mapping: day_risk=day_risk -> day_risk=_PT_DAY_RISK
# Only patch inside the first ctx = { ... } block after "pt_run_loop(" marker.
$src2 = $src
# Simple replace for the common pattern:
$src2 = $src2 -replace '(?m)^\s*day_risk\s*=\s*day_risk\s*,\s*$', '        day_risk=_PT_DAY_RISK,'
# Also handle dict-literal form inside ctx:
$src2 = $src2 -replace '(?m)^\s*"day_risk"\s*:\s*day_risk\s*,\s*$', '        "day_risk": _PT_DAY_RISK,'

if ($src2 -ne $src) {
  $src = $src2
  Write-Host "[OK] Patched cutover ctx: day_risk=_PT_DAY_RISK" -ForegroundColor Green
} else {
  Write-Host "[WARN] Did not find day_risk mapping to replace (may already be fixed or different format)." -ForegroundColor Yellow
}

Set-Content -Path $paper -Value $src -Encoding UTF8

# 3) Compile check
$py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

& $py -m py_compile $paper
if ($LASTEXITCODE -ne 0) { throw "py_compile failed with exit code $LASTEXITCODE" }

Write-Host "[OK] Compile: paper_trader.py" -ForegroundColor Green
