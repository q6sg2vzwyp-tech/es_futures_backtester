param(
  [string]$RepoRoot = (Get-Location).Path
)

$ErrorActionPreference = "Stop"

$target = Join-Path $RepoRoot "paper_trader.py"
if (!(Test-Path $target)) { throw "Cannot find $target. Run from repo root." }

$src = Get-Content $target -Raw -Encoding UTF8

if ($src -match "PT_CUTOVER_TO_PT_LOOP") {
  Write-Host "[OK] paper_trader.py already contains PT_CUTOVER_TO_PT_LOOP block. No changes." -ForegroundColor Green
  exit 0
}

# Find "# MAIN LOOP" marker
$marker = [regex]::Match($src, "(?m)^(?<indent>\s*)#\s*MAIN\s+LOOP\s*$")
if (!$marker.Success) {
  throw "Could not find '# MAIN LOOP' marker in paper_trader.py. Aborting (safe)."
}

$indent = $marker.Groups["indent"].Value

# Insert right after marker line
$insertBlock = @"
${indent}# ================== PT_CUTOVER_TO_PT_LOOP v1 ==================
${indent}# Delegates the canonical loop to pt.loop_core.run_loop(ctx) and RETURNS.
${indent}# Legacy while True loop below remains as a rollback safety net but is unreachable.
${indent}try:
${indent}    import pt.loop_core as _pt_loop_core
${indent}except Exception as _e:
${indent}    _pt_loop_core = None
${indent}
${indent}class _PTLoggerShim:
${indent}    def info(self, msg, *a):
${indent}        try:
${indent}            s = (msg % a) if a else str(msg)
${indent}        except Exception:
${indent}            s = str(msg)
${indent}        try:
${indent}            log("pt_info", msg=s)
${indent}        except Exception:
${indent}            pass
${indent}
${indent}    def warning(self, msg, *a):
${indent}        try:
${indent}            s = (msg % a) if a else str(msg)
${indent}        except Exception:
${indent}            s = str(msg)
${indent}        try:
${indent}            log("pt_warn", msg=s)
${indent}        except Exception:
${indent}            pass
${indent}
${indent}    def exception(self, msg, *a):
${indent}        # best-effort traceback passthrough
${indent}        try:
${indent}            import traceback as _tb
${indent}            tb = _tb.format_exc()
${indent}        except Exception:
${indent}            tb = ""
${indent}        try:
${indent}            s = (msg % a) if a else str(msg)
${indent}        except Exception:
${indent}            s = str(msg)
${indent}        try:
${indent}            log("pt_exc", msg=s, tb=tb)
${indent}        except Exception:
${indent}            pass
${indent}
${indent}_pt_logger = _PTLoggerShim()
${indent}
${indent}if _pt_loop_core is not None:
${indent}    _pt_ctx = {
${indent}        "args": args,
${indent}        "logger": _pt_logger,
${indent}        "ib": ib,
${indent}        "con": con,
${indent}        "ticker": ticker,
${indent}        "bars": C,
${indent}        "day_risk": day_risk,
${indent}        "week_state": week_state,
${indent}        "bandit": bandit,
${indent}        "meta": meta,
${indent}        "shadow": shadow,
${indent}    }
${indent}    try:
${indent}        _pt_loop_core.run_loop(_pt_ctx)
${indent}    finally:
${indent}        # always return to prevent legacy loop from running
${indent}        return
${indent}# ================== END PT_CUTOVER_TO_PT_LOOP ==================

"@

# compute insertion index: end of marker line
$lineEnd = $src.IndexOf("`n", $marker.Index)
if ($lineEnd -lt 0) { $lineEnd = $marker.Index + $marker.Length }
else { $lineEnd = $lineEnd + 1 } # include newline

# Backup
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$bak = "$target.bak_cutover_$ts"
Copy-Item $target $bak -Force
Write-Host "[BACKUP] $bak" -ForegroundColor Yellow

# Write
$new = $src.Substring(0, $lineEnd) + $insertBlock + $src.Substring($lineEnd)
Set-Content -Path $target -Value $new -Encoding UTF8
Write-Host "[OK] Inserted PT cutover delegation block in paper_trader.py" -ForegroundColor Green
Write-Host "     Legacy while True loop below remains but is unreachable (return before it)." -ForegroundColor Green

