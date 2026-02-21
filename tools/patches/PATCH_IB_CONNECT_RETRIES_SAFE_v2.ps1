<# 
PATCH_IB_CONNECT_RETRIES_SAFE_v2.ps1

Purpose (safe, minimal, retry-aware):
1) Ensure pt\ib_connect.py contains helper: connect_existing_ib_from_args(...)
   - Removes any stray "from __future__ import annotations" lines (fixes SyntaxError)
2) Update paper_trader.py connect_with_retries to use the helper while preserving indentation:
   ib.connect(args.host, args.port, clientId=cid, timeout=args.connect_timeout_sec)
   -> connect_existing_ib_from_args(ib, args, client_id=cid, logger=log)

Safety:
- Refuses to run if paper_trader.py is suspiciously small (< 100000 bytes)
- Creates timestamped backups in .\backups\
- Atomic writes
#>

$ErrorActionPreference = "Stop"

function Get-Root {
  if ($PSScriptRoot) { return (Resolve-Path (Join-Path $PSScriptRoot "..")).Path }
  return (Resolve-Path ".").Path
}

function Ensure-Dir($p) {
  if (-not (Test-Path $p)) { New-Item -ItemType Directory -Path $p | Out-Null }
}

function Write-AtomicText([string]$path, [string]$text) {
  $tmp = "$path.tmp__$(Get-Random)"
  [System.IO.File]::WriteAllText($tmp, $text, [System.Text.Encoding]::UTF8)
  Move-Item -Force $tmp $path
}

function Backup-File([string]$path, [string]$backupDir) {
  $ts = Get-Date -Format "yyyyMMdd_HHmmss"
  $name = Split-Path $path -Leaf
  $dest = Join-Path $backupDir ("{0}_{1}.bak" -f $name, $ts)
  Copy-Item -Force $path $dest
  return $dest
}

$ROOT = Get-Root
$BACKUPS = Join-Path $ROOT "backups"
Ensure-Dir $BACKUPS

$paper = Join-Path $ROOT "paper_trader.py"
$ibmod = Join-Path $ROOT "pt\ib_connect.py"

if (-not (Test-Path $paper)) { throw "Missing: $paper" }
if (-not (Test-Path $ibmod)) { throw "Missing: $ibmod" }

# Guard against truncation
$minBytes = 100000
$paperBytes = (Get-Item $paper).Length
if ($paperBytes -lt $minBytes) {
  throw "ABORT: paper_trader.py size ($paperBytes) < $minBytes bytes. Restore from backups first (.\backups\paper_trader_pre_*.py)."
}

$bkPaper = Backup-File $paper $BACKUPS
$bkIb    = Backup-File $ibmod $BACKUPS
Write-Host "[OK] Backed up paper_trader.py -> $bkPaper"
Write-Host "[OK] Backed up pt\ib_connect.py -> $bkIb"

# --- pt\ib_connect.py: remove stray future import + ensure helper exists ---
$ibText = Get-Content $ibmod -Raw -Encoding UTF8

# Remove any "from __future__ import annotations" lines (they must be at file top; safest is to remove entirely)
$ibText = [regex]::Replace($ibText, "(?m)^\s*from\s+__future__\s+import\s+annotations\s*\r?\n", "")

$helperBegin = "# === CONNECT_EXISTING_HELPER v1 BEGIN ==="
$helperEnd   = "# === CONNECT_EXISTING_HELPER v1 END ==="

$helperBlock = @"
# === CONNECT_EXISTING_HELPER v1 BEGIN ===
from dataclasses import dataclass, replace
from typing import Optional, Any

@dataclass(frozen=True)
class _IBConnSpec:
    host: str = "127.0.0.1"
    port: int = 4002
    client_id: int = 1111
    readonly: bool = True
    timeout: float = 6.0

def _spec_from_args(args: Any) -> _IBConnSpec:
    host = getattr(args, "host", "127.0.0.1") or "127.0.0.1"
    port = int(getattr(args, "port", 4002))
    client_id = int(getattr(args, "clientId", 1111))
    readonly = bool(getattr(args, "readonly", True))
    timeout = float(getattr(args, "connect_timeout_sec", 6.0))
    return _IBConnSpec(host=host, port=port, client_id=client_id, readonly=readonly, timeout=timeout)

def connect_existing_ib_from_args(ib: Any, args: Any, *, client_id: Optional[int] = None, logger: Optional[Any] = None) -> Any:
    spec = _spec_from_args(args)
    if client_id is not None:
        spec = replace(spec, client_id=int(client_id))
    # NOTE: logger is expected to be your existing log(...) function (safe if absent/mismatched)
    if logger:
        try:
            logger("boot_progress", step="connecting", host=spec.host, port=spec.port, clientId=spec.client_id)
        except Exception:
            pass
    try:
        ib.connect(spec.host, spec.port, clientId=spec.client_id, timeout=spec.timeout)
    except TypeError:
        ib.connect(spec.host, spec.port, clientId=spec.client_id)
    return ib
# === CONNECT_EXISTING_HELPER v1 END ===
"@

if ($ibText -match [regex]::Escape($helperBegin) -and $ibText -match [regex]::Escape($helperEnd)) {
  $pattern = [regex]::Escape($helperBegin) + ".*?" + [regex]::Escape($helperEnd)
  $ibText = [regex]::Replace($ibText, $pattern, $helperBlock, "Singleline")
  Write-Host "[OK] Replaced existing connect_existing helper block in pt\ib_connect.py"
} else {
  if (-not $ibText.EndsWith("`n")) { $ibText += "`n" }
  $ibText = $ibText + "`n" + $helperBlock + "`n"
  Write-Host "[OK] Appended connect_existing helper block to pt\ib_connect.py"
}

Write-AtomicText $ibmod $ibText

# --- paper_trader.py: ensure import + rewrite connect line with indentation preserved ---
$paperText = Get-Content $paper -Raw -Encoding UTF8

if ($paperText -notmatch "(?m)^\s*from\s+pt\.ib_connect\s+import\s+connect_existing_ib_from_args") {
  # simple prepend to avoid docstring parsing complexity
  $paperText = "from pt.ib_connect import connect_existing_ib_from_args`r`n" + $paperText
  Write-Host "[OK] Added import: from pt.ib_connect import connect_existing_ib_from_args"
}

$pat = "(?m)^(\s*)ib\.connect\(args\.host,\s*args\.port,\s*clientId=cid,\s*timeout=args\.connect_timeout_sec\)\s*$"
if ($paperText -match $pat) {
  $paperText = [regex]::Replace($paperText, $pat, '${1}connect_existing_ib_from_args(ib, args, client_id=cid, logger=log)', 1)
  Write-Host "[OK] Rewired connect_with_retries connect line -> connect_existing_ib_from_args(... client_id=cid ...)"
} else {
  Write-Host "[WARN] Exact connect line not found; no rewrite applied."
  Write-Host "       Expected: ib.connect(args.host, args.port, clientId=cid, timeout=args.connect_timeout_sec)"
}

Write-AtomicText $paper $paperText

Write-Host ""
Write-Host "Verify compile:"
Write-Host "  .\.venv\Scripts\python.exe -m py_compile .\paper_trader.py .\pt\ib_connect.py"
