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

# If _hb_lock already exists, do nothing
if ($txt -match "(?m)^\s*_hb_lock\s*=") {
  Write-Host "[HOTFIX] _hb_lock already present. No changes made."
  exit 0
}

# Ensure threading import exists (best-effort, don't duplicate)
if ($txt -notmatch "(?m)^\s*import\s+threading\s*$" -and $txt -notmatch "(?m)^\s*from\s+threading\s+import\s+") {
  # Insert 'import threading' after last import line near the top (first 200 lines)
  $lines = $txt -split "`n"
  $insertAt = -1
  for ($i=0; $i -lt [Math]::Min($lines.Count, 250); $i++) {
    if ($lines[$i] -match "^\s*(import|from)\s+") { $insertAt = $i }
  }
  if ($insertAt -ge 0) {
    $lines = @($lines[0..$insertAt] + "import threading" + $lines[($insertAt+1)..($lines.Count-1)])
    $txt = ($lines -join "`n")
    Write-Host "[HOTFIX] inserted: import threading"
  } else {
    # fallback: prepend
    $txt = "import threading`n" + $txt
    Write-Host "[HOTFIX] prepended: import threading"
  }
}

# Heartbeat globals to insert (matches prior defaults)
$hbBlock = @'
# ---------- Heartbeat (thread) ----------
_hb_lock = threading.Lock()
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
    "parent_entry_id": None,
    "parent_to_mkt_limit_sec": None,
    "parent_to_mkt_age_sec": None,
    "parent_to_mkt_remaining_sec": None,
}
'@

# Insert block immediately before def hb_update
$pattern = "(?ms)^\s*def\s+hb_update\s*\("
if ($txt -notmatch $pattern) {
  throw "Could not find 'def hb_update(' in paper_trader.py; cannot apply hotfix safely."
}

$txt = [regex]::Replace($txt, $pattern, ($hbBlock + "`n`n" + "def hb_update("), 1)

# Write back
Set-Content -Path $PT -Value $txt -Encoding UTF8

Write-Host "[HOTFIX] Applied heartbeat globals fix."
Write-Host "[HOTFIX] Backup saved to: $BKDIR"
Write-Host "[HOTFIX] Next: run .\.venv\Scripts\python.exe .\paper_trader.py --help"