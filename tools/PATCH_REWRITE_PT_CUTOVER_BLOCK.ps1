param(
  [string]$RepoRoot = (Get-Location).Path
)
$ErrorActionPreference = "Stop"
Set-Location $RepoRoot

$paper = Join-Path $RepoRoot "paper_trader.py"
if (!(Test-Path $paper)) { throw "Missing: $paper" }

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
Copy-Item $paper "$paper.bak_rewrite_cutover_$ts" -Force
Write-Host "[BACKUP] $paper.bak_rewrite_cutover_$ts" -ForegroundColor DarkGray

$src = Get-Content $paper -Raw -Encoding UTF8

# Find the cutover marker indentation
$m = [regex]::Match($src, '(?m)^(?<ind>\s*)#\s*={2,}\s*PT_CUTOVER_TO_PT_RUN_LOOP\b.*$')
if (-not $m.Success) {
  throw "Could not find PT cutover marker in paper_trader.py (PT_CUTOVER_TO_PT_RUN_LOOP)."
}
$ind = $m.Groups["ind"].Value
$ind2 = $ind + "    "
$ind3 = $ind2 + "    "

# Replace everything from the cutover marker up to (but not including) the legacy 'while True:' loop.
$pattern = '(?ms)^(?<ind>\s*)#\s*={2,}\s*PT_CUTOVER_TO_PT_RUN_LOOP\b.*?^\s*while\s+True\s*:'

$nl = "`r`n"
$block =
  ($ind + "# ================== PT_CUTOVER_TO_PT_RUN_LOOP v4 (rewritten) ==================" + $nl) +
  ($ind + "# Delegates main loop to pt.loop_core.run_loop(ctx)." + $nl) +
  ($ind + "# This block is intentionally defensive: it pulls optional ctx keys from locals() to avoid NameError." + $nl) +
  ($ind + "try:" + $nl) +
  ($ind2 + "from pt.loop_core import run_loop as pt_run_loop" + $nl) +
  ($nl) +
  ($ind2 + "# loop_core expects a live ib_insync Ticker object." + $nl) +
  ($ind2 + "try:" + $nl) +
  ($ind3 + "_pt_ticker = ib.reqMktData(con)" + $nl) +
  ($ind3 + "try:" + $nl) +
  ($ind3 + "    ib.sleep(0.2)" + $nl) +
  ($ind3 + "except Exception:" + $nl) +
  ($ind3 + "    pass" + $nl) +
  ($ind2 + "except Exception:" + $nl) +
  ($ind3 + "_pt_ticker = None" + $nl) +
  ($nl) +
  ($ind2 + "# Build ctx with required keys first" + $nl) +
  ($ind2 + "_pt_ctx = {" + $nl) +
  ($ind3 + '"args": args,' + $nl) +
  ($ind3 + '"logger": log,' + $nl) +
  ($ind3 + '"ib": ib,' + $nl) +
  ($ind3 + '"con": con,' + $nl) +
  ($ind3 + '"ticker": _pt_ticker,' + $nl) +
  ($ind2 + "}" + $nl) +
  ($nl) +
  ($ind2 + "# Optional keys (only if they exist in this main() scope)" + $nl) +
  ($ind2 + "for _k in @(" + $nl) +
  ($ind3 + '"bars","bars_15m","day_risk","week_state","bandit","meta","shadow",' + $nl) +
  ($ind3 + '"margin_mgr","day_policy_state","eod_state","trade_start","trade_end",' + $nl) +
  ($ind3 + '"build_and_write_heartbeat","build_bandit_hb_fields","append_shadow_roundtrip_log",' + $nl) +
  ($ind3 + '"roll_week_if_needed","maybe_daily_restart","is_us_market_holiday",' + $nl) +
  ($ind3 + '"AUTO_FLAT_CT","DAILY_RESTART_CT","DAILY_RESTART_JSON","HB_PATH","RUNTIME_STATE_JSON",' + $nl) +
  ($ind3 + '"IB_ERROR_DECAY_SEC","ORPHAN_SWEEP_COOLDOWN","SHADOW_START_CT","SHADOW_END_CT",' + $nl) +
  ($ind3 + '"BAYES_SOURCE","BAYES_TRAIN_CSV","LEARN_BAYES_BEST","LEARN_MODEL_PATH",' + $nl) +
  ($ind3 + '"TRADE_LOG_CSV","compute_boost_factor","run_eod_bayes_opt_filtered","build_bayes_training_set"' + $nl) +
  ($ind2 + "):" + $nl) +
  ($ind3 + "try:" + $nl) +
  ($ind3 + "    if (Get-Variable -Name _k -Scope Local -ErrorAction SilentlyContinue) {" + $nl) +
  ($ind3 + "        $val = (Get-Variable -Name _k -Scope Local).Value" + $nl) +
  ($ind3 + "        if ($null -ne $val) { _pt_ctx[_k] = $val }" + $nl) +
  ($ind3 + "    }" + $nl) +
  ($ind3 + "except Exception:" + $nl) +
  ($ind3 + "    pass" + $nl) +
  ($nl) +
  ($ind2 + "# Run loop and exit main() (legacy while True remains below but is unreachable)" + $nl) +
  ($ind2 + "rc = pt_run_loop(_pt_ctx)" + $nl) +
  ($ind2 + "return" + $nl) +
  ($ind + "except Exception as e:" + $nl) +
  ($ind2 + "log('pt_cutover_err', err=repr(e))" + $nl) +
  ($ind2 + "# Fall through to legacy loop if cutover fails (useful during migration)" + $nl) +
  ($ind + "# ================== END PT_CUTOVER_TO_PT_RUN_LOOP ==================" + $nl) +
  ($nl) +
  ($ind + "while True:")

$src2 = [regex]::Replace($src, $pattern, $block, 1)
if ($src2 -eq $src) { throw "Failed to rewrite PT cutover block (pattern not matched as expected)." }

Set-Content -Path $paper -Value $src2 -Encoding UTF8

$py = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (!(Test-Path $py)) { $py = "python" }
& $py -m py_compile $paper
if ($LASTEXITCODE -ne 0) { throw "py_compile failed with exit code $LASTEXITCODE" }

Write-Host "[OK] Rewrote PT cutover block + compile OK" -ForegroundColor Green
