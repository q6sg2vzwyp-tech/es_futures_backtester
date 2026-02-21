param(
  [string]$Path = ".\paper_trader.py"
)

$ErrorActionPreference = "Stop"
if (!(Test-Path $Path)) { throw "Not found: $Path" }

# Backup
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$bak = "$Path.bak_fix_v4_weekstate_$ts"
Copy-Item $Path $bak -Force
Write-Host "[BACKUP] $bak" -ForegroundColor Yellow

$src = Get-Content $Path -Raw -Encoding UTF8

# 1) Disable/remove legacy v3 cutover block if present (it causes scope regressions)
$reV3 = '(?s)(^[ \t]*#\s*=+\s*PT_CUTOVER_TO_PT_LOOP v3\s*=+.*?^[ \t]*#\s*=+\s*END PT_CUTOVER_TO_PT_LOOP v3\s*=+\s*\r?\n)'
if ($src -match $reV3) {
  $indent = ([regex]::Match($src, $reV3, [System.Text.RegularExpressions.RegexOptions]::Multiline)).Value
  $indent = ([regex]::Match($indent, '^[ \t]*', [System.Text.RegularExpressions.RegexOptions]::Multiline)).Value
  $rep = $indent + "# [DISABLED] legacy cutover v3 removed (superseded by v4)`r`n" + $indent + "pass`r`n"
  $src = [regex]::Replace($src, $reV3, [System.Text.RegularExpressions.MatchEvaluator]{ param($m) $rep }, [System.Text.RegularExpressions.RegexOptions]::Multiline)
  Write-Host "[OK] Disabled v3 cutover block" -ForegroundColor Green
} else {
  Write-Host "[INFO] No v3 cutover block found" -ForegroundColor DarkGray
}

# 2) Ensure week_state + snapshot exist before v4 ctx build
$guard = @"
                # ---- PT cutover (v4): ensure required ctx vars exist in main() scope ----
                try:
                    snapshot  # type: ignore[name-defined]
                except Exception:
                    snapshot = {}
                try:
                    week_state  # type: ignore[name-defined]
                except Exception:
                    try:
                        # Prefer pt.week_guard helper (already imported in this repo)
                        week_state = init_week_state(
                            float(globals().get("restored_week_R", 0.0) or 0.0),
                            str(globals().get("restored_week_id", "" ) or ""),
                            ct_now().date(),
                            args,
                        )
                    except Exception:
                        try:
                            _wkR = float(globals().get("restored_week_R", 0.0) or 0.0)
                        except Exception:
                            _wkR = 0.0
                        week_state = {
                            "week_R": _wkR,
                            "week_halted": False,
                            "last_week_id": str(globals().get("restored_week_id", "" ) or ""),
                            "weekly_cap_R": float(getattr(args, "weekly_cap_R", 0.0) or 0.0),
                        }
                # ---- END required ctx vars ----

"@

$reCtx = '(?m)^[ \t]*#\s*Build ctx contract\s*\r?\n([ \t]*)ctx\s*=\s*dict\s*\('
if ($src -match $reCtx) {
  $src = [regex]::Replace($src, $reCtx, [System.Text.RegularExpressions.MatchEvaluator]{
    param($m)
    $indent = $m.Groups[1].Value
    # Guard is already indented at 16 spaces; re-indent to match actual ctx indent
    $g = $guard -replace '^(?m) {16}', $indent
    return ($m.Value -replace '(?m)^[ \t]*#\s*Build ctx contract\s*\r?\n', "# Build ctx contract`r`n" + $g)
  }, [System.Text.RegularExpressions.RegexOptions]::Multiline)
  Write-Host "[OK] Inserted week_state+snapshot guard before v4 ctx build" -ForegroundColor Green
} else {
  throw "Could not find v4 ctx build marker '# Build ctx contract' followed by 'ctx = dict('"
}

Set-Content -Path $Path -Value $src -Encoding UTF8

# Compile check
& .\.venv\Scripts\python.exe -m py_compile $Path
Write-Host "[OK] Compile: $Path" -ForegroundColor Green
