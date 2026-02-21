<# 
PATCH_IB_BOOTSTRAP_EXTRACT_SAFE_v2.ps1

Fixes v1 issue:
- v1 could replace an indented ib.connect(...) line with an unindented bootstrap call, breaking a try: block.
- v2 upgrades the injected pt\ib_connect.py block and patches ONLY the ib.connect(...) line using preserved indentation.

Run:
  cd C:\Users\owner\Desktop\es_futures_backtester
  powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\PATCH_IB_BOOTSTRAP_EXTRACT_SAFE_v2.ps1
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
  throw "ABORT: paper_trader.py size ($paperBytes) < $minBytes bytes. Restore from backups first."
}

$bkPaper = Backup-File $paper $BACKUPS
$bkIb    = Backup-File $ibmod $BACKUPS
Write-Host "[OK] Backed up paper_trader.py -> $bkPaper"
Write-Host "[OK] Backed up pt\ib_connect.py -> $bkIb"

# Upgrade/replace IB block in pt\ib_connect.py (replace v1 or v2 to v2)
$begin1 = "# === IB_BOOTSTRAP_EXTRACT v1 BEGIN ==="
$end1   = "# === IB_BOOTSTRAP_EXTRACT v1 END ==="
$begin2 = "# === IB_BOOTSTRAP_EXTRACT v2 BEGIN ==="
$end2   = "# === IB_BOOTSTRAP_EXTRACT v2 END ==="

$inject = @'
# === IB_BOOTSTRAP_EXTRACT v2 BEGIN ===
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Any

@dataclass(frozen=True)
class IBConnSpec:
    host: str = "127.0.0.1"
    port: int = 4002
    client_id: int = 1111
    readonly: bool = True
    timeout: float = 6.0
    account: Optional[str] = None

def _coerce_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default

def _coerce_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return default

def spec_from_args(args: Any, *, defaults: Optional[IBConnSpec] = None) -> IBConnSpec:
    d = defaults or IBConnSpec()
    host = getattr(args, "host", d.host) or d.host
    port = _coerce_int(getattr(args, "port", d.port), d.port)
    client_id = _coerce_int(getattr(args, "clientId", None) or getattr(args, "client_id", d.client_id), d.client_id)
    readonly = bool(getattr(args, "readonly", d.readonly))
    timeout = _coerce_float(getattr(args, "timeout", None) or getattr(args, "connect_timeout_sec", d.timeout), d.timeout)
    account = getattr(args, "account", None) or getattr(args, "ib_account", None) or d.account
    return IBConnSpec(host=host, port=port, client_id=client_id, readonly=readonly, timeout=timeout, account=account)

def _connect(ib: Any, spec: IBConnSpec, *, logger: Optional[Any] = None) -> Any:
    if logger:
        logger.info(f"[IB] Connecting host={spec.host} port={spec.port} clientId={spec.client_id} readonly={spec.readonly} timeout={spec.timeout}")
    try:
        ib.connect(spec.host, spec.port, clientId=spec.client_id, readonly=spec.readonly, timeout=spec.timeout)
    except TypeError:
        ib.connect(spec.host, spec.port, clientId=spec.client_id, readonly=spec.readonly)
    if spec.account and logger:
        logger.info(f"[IB] Account preference supplied: {spec.account}")
    return ib

def bootstrap_ib(spec: IBConnSpec, *, logger: Optional[Any] = None, connect_fn: Optional[Any] = None) -> Any:
    try:
        from ib_insync import IB  # type: ignore
    except Exception as e:
        raise RuntimeError("ib_insync is required for IB bootstrap but could not be imported") from e
    ib = IB() if connect_fn is None else connect_fn()
    _connect(ib, spec, logger=logger)
    if logger:
        try:
            ok = bool(getattr(ib, "isConnected")() if callable(getattr(ib, "isConnected", None)) else getattr(ib, "isConnected", False))
            logger.info(f"[IB] Connected={ok}")
        except Exception:
            logger.info("[IB] Connected (status unknown)")
    return ib

def bootstrap_ib_from_args(args: Any, *, logger: Optional[Any] = None) -> Any:
    return bootstrap_ib(spec_from_args(args), logger=logger)

def connect_existing_ib_from_args(
    ib: Any,
    args: Any,
    *,
    client_id: Optional[int] = None,
    logger: Optional[Any] = None,
) -> Any:
    """Connect an already-created IB() instance using args, with optional clientId override."""
    spec = spec_from_args(args)
    if client_id is not None:
        spec = replace(spec, client_id=int(client_id))
    return _connect(ib, spec, logger=logger)

# === IB_BOOTSTRAP_EXTRACT v2 END ===

'@

$ibText = Get-Content $ibmod -Raw -Encoding UTF8

if ($ibText -match [regex]::Escape($begin2) -and $ibText -match [regex]::Escape($end2)) {
  $pattern = [regex]::Escape($begin2) + ".*?" + [regex]::Escape($end2)
  $ibNew = [regex]::Replace($ibText, $pattern, $inject, "Singleline")
  Write-Host "[OK] Replaced existing v2 block in pt\ib_connect.py"
} elseif ($ibText -match [regex]::Escape($begin1) -and $ibText -match [regex]::Escape($end1)) {
  $pattern = [regex]::Escape($begin1) + ".*?" + [regex]::Escape($end1)
  $ibNew = [regex]::Replace($ibText, $pattern, $inject, "Singleline")
  Write-Host "[OK] Upgraded v1 block -> v2 in pt\ib_connect.py"
} else {
  if (-not $ibText.EndsWith("`n")) { $ibText += "`n" }
  $ibNew = $ibText + "`n" + $inject + "`n"
  Write-Host "[OK] Appended v2 block to pt\ib_connect.py"
}

Write-AtomicText $ibmod $ibNew

# Patch paper_trader.py: ensure import + replace connect line with indentation preserved
$paperText = Get-Content $paper -Raw -Encoding UTF8
$changed = $false

if ($paperText -notmatch "(?m)^\s*from\s+pt\.ib_connect\s+import\s+connect_existing_ib_from_args") {
  # Insert near the top (after docstring/import header region)
  $head = $paperText.Substring(0, [Math]::Min(4000, $paperText.Length))
  $m = [regex]::Match($head, "(?ms)^(.*?)(\r?\n\r?\n)", "Singleline")
  if ($m.Success) {
    $insertAt = $m.Groups[1].Length
    $paperText = $paperText.Insert($insertAt, "`r`nfrom pt.ib_connect import connect_existing_ib_from_args`r`n")
  } else {
    $paperText = "from pt.ib_connect import connect_existing_ib_from_args`r`n" + $paperText
  }
  $changed = $true
  Write-Host "[OK] Added import: from pt.ib_connect import connect_existing_ib_from_args"
}

# Replace the specific connect line used in your connect_with_retries block:
#   ib.connect(args.host, args.port, clientId=cid, timeout=args.connect_timeout_sec)
$pat = "(?m)^(\s*)ib\.connect\(args\.host,\s*args\.port,\s*clientId=cid,\s*timeout=args\.connect_timeout_sec\)\s*$"
if ($paperText -match $pat) {
  $paperText = [regex]::Replace(
    $paperText,
    $pat,
    '${1}connect_existing_ib_from_args(ib, args, client_id=cid, logger=log)',
    1
  )
  $changed = $true
  Write-Host "[OK] Rewired connect_with_retries connect line -> connect_existing_ib_from_args(... client_id=cid ...)"
} else {
  Write-Host "[WARN] Did not find exact connect line pattern in paper_trader.py. No connect rewrite applied."
  Write-Host "       Search for: ib.connect(args.host, args.port, clientId=cid, timeout=args.connect_timeout_sec)"
}

if ($changed) {
  Write-AtomicText $paper $paperText
  Write-Host "[OK] paper_trader.py updated."
}

Write-Host ""
Write-Host "Verify compile:"
Write-Host "  .\.venv\Scripts\python.exe -m py_compile .\paper_trader.py .\pt\ib_connect.py"
