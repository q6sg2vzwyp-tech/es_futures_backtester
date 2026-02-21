param(
  [string]$Target = ".\paper_trader.py"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (!(Test-Path $Target)) { throw "Target not found: $Target" }

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$bak = "$Target.bak_fix_weekstate_snapshot_$ts"
Copy-Item $Target $bak -Force
Write-Host "[BACKUP] $bak" -ForegroundColor Yellow

$src = Get-Content $Target -Raw -Encoding UTF8

# We patch inside the PT_ENABLE_CUTOVER block, right before the ctx dict is built.
# Insert once only if the marker isn't present.
$marker = "# --- PT cutover: ensure week_state + snapshot exist in main scope ---"
if ($src -notmatch [regex]::Escape($marker)) {

  $insert = @"
$marker
        # week_state is created inside the legacy loop; cutover runs before that.
        try:
            week_state  # type: ignore[name-defined]
        except Exception:
            try:
                # prefer pt.week_guard init (already imported as init_week_state)
                week_state = init_week_state(
                    float(locals().get("restored_week_R", 0.0)),
                    str(locals().get("restored_week_id", "")),
                    ct_now().date(),
                    args,
                )
                log("pt_cutover_week_state_init", week_R=float(getattr(week_state, "get", lambda k, d=None: week_state[k] if k in week_state else d)("week_R", 0.0)))
            except Exception as _e:
                week_state = {"week_R": 0.0, "week_halted": False, "last_week_id": "", "weekly_cap_R": 0.0}
                try:
                    log("pt_cutover_week_state_fallback", err=repr(_e))
                except Exception:
                    pass

        # snapshot is used by decision pipeline; ensure it exists outside legacy loop.
        try:
            snapshot  # type: ignore[name-defined]
        except Exception:
            snapshot = {}
"@

  # Find the first occurrence of "ctx = dict(" within the cutover region and insert immediately before it.
  $rx = [regex]::new("(?ms)^(\s*)ctx\s*=\s*dict\s*\(", "Multiline")
  $m = $rx.Match($src)
  if (!$m.Success) { throw "Could not find 'ctx = dict(' to patch. File layout unexpected." }

  $indent = $m.Groups[1].Value
  # Ensure inserted block uses same base indentation
  $insertIndented = $insert -replace "(?m)^", $indent

  $src = $src.Substring(0, $m.Index) + $insertIndented + "`n" + $src.Substring($m.Index)
  Write-Host "[OK] Inserted week_state + snapshot guard before ctx build" -ForegroundColor Green
} else {
  Write-Host "[OK] Marker already present; no insert needed" -ForegroundColor Green
}

# Sanity: if ctx passes week_state=week_state but week_state is still not defined somewhere else,
# this patch ensures it exists before ctx build. No further changes required.

Set-Content -Path $Target -Value $src -Encoding UTF8

# Compile check
$py = ".\.venv\Scripts\python.exe"
if (Test-Path $py) {
  & $py -m py_compile $Target
  Write-Host "[OK] Compile: $Target" -ForegroundColor Green
} else {
  Write-Host "[WARN] .venv python not found; skipped compile" -ForegroundColor Yellow
}
