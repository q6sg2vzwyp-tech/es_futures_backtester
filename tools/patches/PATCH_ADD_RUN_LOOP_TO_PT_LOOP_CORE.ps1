param(
  [string]$RepoRoot = (Get-Location).Path
)

$ErrorActionPreference = "Stop"

$target = Join-Path $RepoRoot "pt\loop_core.py"
if (!(Test-Path $target)) {
  throw "Cannot find $target. Run from repo root (es_futures_backtester)."
}

$src = Get-Content $target -Raw -Encoding UTF8

if ($src -match "(?m)^\s*def\s+run_loop\s*\(") {
  Write-Host "[OK] pt\loop_core.py already has def run_loop(...). No changes." -ForegroundColor Green
  exit 0
}

# Find insertion point right before run_loop_iteration
$pat = "(?m)^\s*def\s+run_loop_iteration\s*\("
$m = [regex]::Match($src, $pat)
if (!$m.Success) {
  throw "Could not find def run_loop_iteration(...) in pt\loop_core.py. Aborting (safe)."
}

$insert = @"
def _shutdown_flag_path() -> str:
    return os.path.join(".", "run", "SHUTDOWN.flag")


def _shutdown_requested() -> bool:
    try:
        return os.path.exists(_shutdown_flag_path())
    except Exception:
        return False


def run_loop(ctx: Dict[str, Any]) -> int:
    """
    Canonical main loop runner for ES Paper Trader (pt architecture).

    This wrapper owns the infinite loop and delegates all strategy/execution/risk decisions
    to run_loop_iteration(ctx).

    Returns process exit code (0 = clean shutdown).
    """
    args = ctx.get("args")
    logger = ctx.get("logger")
    ib = ctx.get("ib")

    # sane defaults
    sleep_sec = float(getattr(args, "loop_sleep_sec", 0.25) or 0.25)
    shutdown_sleep_sec = float(getattr(args, "shutdown_poll_sleep_sec", 0.25) or 0.25)

    if logger:
        try:
            logger.info("[loop] run_loop start (sleep=%.3fs)", sleep_sec)
        except Exception:
            pass

    while True:
        # external shutdown flag (matches STOP scripts)
        if _shutdown_requested() or bool(ctx.get("shutdown", False)):
            if logger:
                try:
                    logger.info("[loop] shutdown requested -> exiting")
                except Exception:
                    pass
            # optional flatten-on-exit if configured and ctx has ib/con
            try:
                if getattr(args, "flatten_on_shutdown", False):
                    ib_local = ctx.get("ib")
                    con_local = ctx.get("con")
                    if ib_local is not None and con_local is not None:
                        flatten_until_flat(ib_local, con_local, logger=logger, note="[shutdown]")
            except Exception:
                pass
            return 0

        try:
            ctx = run_loop_iteration(ctx) or ctx
        except SystemExit:
            raise
        except Exception as e:
            if logger:
                try:
                    logger.exception("[loop] iteration error: %s", e)
                except Exception:
                    pass
            # backoff to avoid hot crash loops
            try:
                if ib is not None:
                    ib.sleep(1.0)
                else:
                    time.sleep(1.0)
            except Exception:
                try:
                    time.sleep(1.0)
                except Exception:
                    pass

        # cadence
        try:
            if ib is not None:
                ib.sleep(sleep_sec)
            else:
                time.sleep(sleep_sec)
        except Exception:
            try:
                time.sleep(shutdown_sleep_sec)
            except Exception:
                pass


"@

# Backup first
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$bak = "$target.bak_pre_runloop_$ts"
Copy-Item $target $bak -Force
Write-Host "[BACKUP] $bak" -ForegroundColor Yellow

# Insert
$pos = $m.Index
$new = $src.Substring(0, $pos) + $insert + $src.Substring($pos)

# Write
Set-Content -Path $target -Value $new -Encoding UTF8
Write-Host "[OK] Patched pt\loop_core.py (added run_loop wrapper)" -ForegroundColor Green
