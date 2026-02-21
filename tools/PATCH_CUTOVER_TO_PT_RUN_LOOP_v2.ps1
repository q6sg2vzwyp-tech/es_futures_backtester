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
Copy-Item $paper "$paper.bak_cutover_pt_runloop_$ts" -Force
Write-Host "[BACKUP] $paper.bak_cutover_pt_runloop_$ts" -ForegroundColor DarkGray

$src = Get-Content $paper -Raw -Encoding UTF8

# 1) Add import: from pt.loop_core import run_loop as pt_run_loop (no backticks)
$importLine = "from pt.loop_core import run_loop as pt_run_loop" + [Environment]::NewLine
if ($src -notmatch '(?m)^\s*from\s+pt\.loop_core\s+import\s+run_loop\s+as\s+pt_run_loop\s*$') {
  # Insert after other pt.* imports if possible, else after typing imports
  if ($src -match '(?m)^\s*from\s+pt\.[a-zA-Z0-9_]+\s+import\s+.*$') {
    $src = [regex]::Replace($src, '(?m)^(?<L>\s*from\s+pt\.[a-zA-Z0-9_]+\s+import\s+.*)$', '${L}' + [Environment]::NewLine + $importLine, 1)
  } else {
    $src = $importLine + $src
  }
  Write-Host "[OK] Added import: pt.loop_core run_loop as pt_run_loop" -ForegroundColor Green
} else {
  Write-Host "[OK] Import already present" -ForegroundColor Green
}

# 2) Insert cutover block right BEFORE the legacy while True loop (first occurrence)
$marker = '(?m)^\s*while\s+True\s*:\s*$'
if ($src -notmatch $marker) {
  throw "Could not find legacy 'while True:' loop to cut over."
}

$cut = @"
    # ================== PT_CUTOVER_TO_PT_LOOP v2 ==================
    # Delegates the main loop to pt.loop_core.run_loop(ctx).
    # Legacy monolithic while-loop remains below but is unreachable after this return.
    try:
        ctx = dict(
            args=args,
            logger=logger,
            ib=ib,
            con=con,
            ticker=ticker,
            bars=bars,
            # Provide any additional ctx keys you already build earlier in paper_trader.py:
            day_risk=day_risk,
            week_state=week_state,
            bandit=learner,
            meta=meta,
            shadow=shadow,
            build_and_write_heartbeat=build_and_write_heartbeat,
            build_bandit_hb_fields=build_bandit_hb_fields,
            margin_mgr=margin_mgr,
            is_us_market_holiday=is_us_market_holiday,
            roll_week_if_needed=roll_week_if_needed,
            maybe_daily_restart=maybe_daily_restart,
            trade_start=trade_start,
            trade_end=trade_end,
            # paths / constants (if present)
            HB_PATH=HB_PATH,
            TRADE_LOG_CSV=TRADE_LOG_CSV,
            RUNTIME_STATE_JSON=RUNTIME_STATE_JSON,
            DAILY_RESTART_JSON=DAILY_RESTART_JSON,
            DAILY_RESTART_CT=DAILY_RESTART_CT,
            AUTO_FLAT_CT=AUTO_FLAT_CT,
            SHADOW_START_CT=SHADOW_START_CT,
            SHADOW_END_CT=SHADOW_END_CT,
            IB_ERROR_DECAY_SEC=IB_ERROR_DECAY_SEC,
            ORPHAN_SWEEP_COOLDOWN=ORPHAN_SWEEP_COOLDOWN,
            STATE_SAVE_EVERY_SEC=STATE_SAVE_EVERY_SEC,
        )
        return int(pt_run_loop(ctx))
    except SystemExit:
        raise
    except Exception as e:
        try:
            log("pt_cutover_err", err=repr(e))
        except Exception:
            pass
        raise
    # ================== END PT_CUTOVER_TO_PT_LOOP v2 ==================
"@

# Insert at the same indent level as while True (usually 8 spaces inside main)
# We'll inject the block immediately before the first while True line.
$src = [regex]::Replace($src, $marker, ($cut + [Environment]::NewLine + '        while True:'), 1)

Set-Content -Path $paper -Value $src -Encoding UTF8
Write-Host "[OK] Inserted cutover block above legacy while loop" -ForegroundColor Green

# 3) Compile check (fail hard)
$py = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (!(Test-Path $py)) { $py = "python" }

& $py -m py_compile $paper $loop
if ($LASTEXITCODE -ne 0) { throw "py_compile failed with exit code $LASTEXITCODE" }
Write-Host "[OK] Compile: paper_trader.py + pt\loop_core.py" -ForegroundColor Green

Write-Host ""
Write-Host "[NEXT] Start with: cmd /c .\tools\START_TRADER.cmd (or your normal start)" -ForegroundColor Cyan
