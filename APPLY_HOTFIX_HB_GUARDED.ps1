Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ROOT = (Resolve-Path ".").Path
$PT = Join-Path $ROOT "paper_trader.py"
if (!(Test-Path $PT)) { throw "paper_trader.py not found in $ROOT" }

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$BKDIR = Join-Path $ROOT ("tools\patches_quarantine_" + $ts)
New-Item -ItemType Directory -Force -Path $BKDIR | Out-Null
Copy-Item $PT (Join-Path $BKDIR "paper_trader.py.BAK") -Force

$txt = Get-Content $PT -Raw -Encoding UTF8

# Ensure threading import exists (insert 'import threading' if missing)
$hasThreadingImport = ($txt -match "(?m)^\s*import\s+threading\s*$") -or ($txt -match "(?m)^\s*from\s+threading\s+import\s+")
if (-not $hasThreadingImport) {
  $lines = $txt -split "`n"
  $insertAt = -1
  for ($i=0; $i -lt [Math]::Min($lines.Count, 300); $i++) {
    if ($lines[$i] -match "^\s*(import|from)\s+") { $insertAt = $i }
  }
  if ($insertAt -ge 0) {
    $lines = @($lines[0..$insertAt] + "import threading" + $lines[($insertAt+1)..($lines.Count-1)])
    $txt = ($lines -join "`n")
    Write-Host "[HOTFIX] inserted: import threading"
  } else {
    $txt = "import threading`n" + $txt
    Write-Host "[HOTFIX] prepended: import threading"
  }
}

# Guarded init: define _hb_lock/_hb_state only if missing.
$hbGuard = @'
# --- HOTFIX: ensure heartbeat globals exist (guarded, no override) ---
try:
    _hb_lock
except NameError:
    _hb_lock = threading.Lock()
try:
    _hb_state
except NameError:
    _hb_state = {
        "state": "-",
        "idle_reason": "starting_or_quiet",
        "net_qty": 0,
        "bars": 0,
        "rt_enabled": False,
        "rt_status": "disabled",
        "rt_age_sec": None,
        "rt_queue_len": 0,
        "in_session_window": False,
        "caps": [],
        "news_kill": False,
        "dayR": 0.0,
        "trades_today": 0,
        "cool_until": None,
        "orders_disabled_paper_safety": False,
        "orders_disabled_paper_safety": False,
        "parent_entry_id": None,
        "parent_to_mkt_limit_sec": None,
        "parent_to_mkt_age_sec": None,
        "parent_to_mkt_remaining_sec": None,
    }
# --- END HOTFIX ---
'@

if ($txt -match "(?m)^\s*# --- HOTFIX: ensure heartbeat globals exist") {
  Write-Host "[HOTFIX] Heartbeat guarded init already present. No changes made."
  exit 0
}

$pattern = "(?ms)^\s*def\s+hb_update\s*\("
if ($txt -notmatch $pattern) {
  throw "Could not find 'def hb_update(' in paper_trader.py; cannot apply hotfix."
}

$txt = [regex]::Replace($txt, $pattern, ($hbGuard + "`n`n" + "def hb_update("), 1)

Set-Content -Path $PT -Value $txt -Encoding UTF8
Write-Host "[HOTFIX] Applied guarded heartbeat init (pre-hb_update)."
Write-Host "[HOTFIX] Backup saved to: $BKDIR"