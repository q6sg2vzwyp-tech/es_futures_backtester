param(
  [string]$RepoRoot = (Get-Location).Path
)

$ErrorActionPreference = "Stop"
Set-Location $RepoRoot

$paper = Join-Path $RepoRoot "paper_trader.py"
$loop  = Join-Path $RepoRoot "pt\loop_core.py"
if (!(Test-Path $paper)) { throw "Missing: $paper" }
if (!(Test-Path $loop))  { throw "Missing: $loop" }

# Ensure pt.loop_core has run_loop(ctx) wrapper
$lc = Get-Content $loop -Raw -Encoding UTF8
if ($lc -notmatch '(?m)^\s*def\s+run_loop\s*\(\s*ctx\s*:\s*Dict\[str,\s*Any\]\s*\)\s*->\s*int\s*:\s*$') {
  throw "pt\loop_core.py does not appear to define: def run_loop(ctx: Dict[str, Any]) -> int:"
}

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
Copy-Item $paper "$paper.bak_cutover_pt_runloop_v3_$ts" -Force
Write-Host "[BACKUP] $paper.bak_cutover_pt_runloop_v3_$ts" -ForegroundColor DarkGray

$src = Get-Content $paper -Raw -Encoding UTF8

# 1) Add import in a SAFE top-level location (anchor on a top-level import line)
# We specifically insert after the existing decision pipeline import at column 0
$needImport = ($src -notmatch '(?m)^from\s+pt\.loop_core\s+import\s+run_loop\s+as\s+pt_run_loop\s*$')
if ($needImport) {
  $anchor = '(?m)^(from\s+pt\.decision_pipeline\s+import\s+decide_and_maybe_place_entry\s*)$'
  if ($src -match $anchor) {
    $src = [regex]::Replace(
      $src,
      $anchor,
      ('$1' + [Environment]::NewLine + 'from pt.loop_core import run_loop as pt_run_loop'),
      1
    )
    Write-Host "[OK] Added import after pt.decision_pipeline import" -ForegroundColor Green
  } else {
    # fallback: after first top-level "from pt.ai_hooks import AIHooks" (column 0)
    $anchor2 = '(?m)^(from\s+pt\.ai_hooks\s+import\s+AIHooks\s*)$'
    if ($src -match $anchor2) {
      $src = [regex]::Replace(
        $src,
        $anchor2,
        ('$1' + [Environment]::NewLine + 'from pt.loop_core import run_loop as pt_run_loop'),
        1
      )
      Write-Host "[OK] Added import after pt.ai_hooks import" -ForegroundColor Green
    } else {
      throw "Could not find a safe top-level anchor import (pt.decision_pipeline or pt.ai_hooks). Aborting."
    }
  }
} else {
  Write-Host "[OK] Import already present: pt_run_loop" -ForegroundColor Green
}

# 2) Insert cutover block right BEFORE the first legacy while True loop line
$marker = '(?m)^(?<indent>\s*)while\s+True\s*:\s*$'
$m = [regex]::Match($src, $marker)
if (-not $m.Success) { throw "Could not find legacy 'while True:' loop to cut over." }

$indent = $m.Groups["indent"].Value
# Guard: we expect this loop to be inside main(), so indent should be >= 4 spaces.
if ($indent.Length -lt 4) { throw "Refusing to patch: 'while True:' appears to be top-level (indent=$($indent.Length))." }

# If already cut over, don't double insert
if ($src -match 'PT_CUTOVER_TO_PT_LOOP v3') {
  Write-Host "[OK] Cutover block already present (v3). Skipping insert." -ForegroundColor Yellow
} else {
  $cut = @"
${indent}# ================== PT_CUTOVER_TO_PT_LOOP v3 ==================
${indent}# Delegates the main loop to pt.loop_core.run_loop(ctx).
${indent}# Legacy monolithic while-loop remains below but is unreachable after this return.
${indent}try:
${indent}    ctx = dict(
${indent}        args=args,
${indent}        logger=logger,
${indent}        ib=ib,
${indent}        con=con,
${indent}        ticker=ticker,
${indent}        bars=bars,
${indent}        day_risk=day_risk,
${indent}        week_state=week_state,
${indent}        bandit=learner,
${indent}        meta=meta,
${indent}        shadow=shadow,
${indent}        build_and_write_heartbeat=build_and_write_heartbeat,
${indent}        build_bandit_hb_fields=build_bandit_hb_fields,
${indent}        margin_mgr=margin_mgr,
${indent}        is_us_market_holiday=is_us_market_holiday,
${indent}        roll_week_if_needed=roll_week_if_needed,
${indent}        maybe_daily_restart=maybe_daily_restart,
${indent}        trade_start=trade_start,
${indent}        trade_end=trade_end,
${indent}        HB_PATH=HB_PATH,
${indent}        TRADE_LOG_CSV=TRADE_LOG_CSV,
${indent}        RUNTIME_STATE_JSON=RUNTIME_STATE_JSON,
${indent}        DAILY_RESTART_JSON=DAILY_RESTART_JSON,
${indent}        DAILY_RESTART_CT=DAILY_RESTART_CT,
${indent}        AUTO_FLAT_CT=AUTO_FLAT_CT,
${indent}        SHADOW_START_CT=SHADOW_START_CT,
${indent}        SHADOW_END_CT=SHADOW_END_CT,
${indent}        IB_ERROR_DECAY_SEC=IB_ERROR_DECAY_SEC,
${indent}        ORPHAN_SWEEP_COOLDOWN=ORPHAN_SWEEP_COOLDOWN,
${indent}        STATE_SAVE_EVERY_SEC=STATE_SAVE_EVERY_SEC,
${indent}    )
${indent}    return int(pt_run_loop(ctx))
${indent}except SystemExit:
${indent}    raise
${indent}except Exception as e:
${indent}    try:
${indent}        log("pt_cutover_err", err=repr(e))
${indent}    except Exception:
${indent}        pass
${indent}    raise
${indent}# ================== END PT_CUTOVER_TO_PT_LOOP v3 ==================
"@

  # Replace the first while True line with cutover + original while True line
  $src = [regex]::Replace($src, $marker, ($cut + [Environment]::NewLine + '${indent}while True:'), 1)
  Write-Host "[OK] Inserted cutover block above legacy while loop" -ForegroundColor Green
}

Set-Content -Path $paper -Value $src -Encoding UTF8

# 3) Compile check (fail hard)
$py = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (!(Test-Path $py)) { $py = "python" }

& $py -m py_compile $paper $loop
if ($LASTEXITCODE -ne 0) { throw "py_compile failed with exit code $LASTEXITCODE" }
Write-Host "[OK] Compile: paper_trader.py + pt\loop_core.py" -ForegroundColor Green
