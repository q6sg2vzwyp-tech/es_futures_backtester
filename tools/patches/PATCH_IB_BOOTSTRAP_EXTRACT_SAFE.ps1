<# 
PATCH_IB_BOOTSTRAP_EXTRACT_SAFE.ps1

Goal:
- Safely inject a centralized IB bootstrap wrapper into pt\ib_connect.py
- Safely update paper_trader.py to call the wrapper (best-effort pattern replace)
- Always create backups and refuse to proceed if paper_trader.py looks truncated.
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
$TOOLS = Join-Path $ROOT "tools"
Ensure-Dir $BACKUPS
Ensure-Dir $TOOLS

$paper = Join-Path $ROOT "paper_trader.py"
$ibmod = Join-Path $ROOT "pt\ib_connect.py"

if (-not (Test-Path $paper)) { throw "Missing: $paper" }
if (-not (Test-Path $ibmod)) { throw "Missing: $ibmod" }

# --- Guard: paper_trader.py size sanity ---
$minBytes = 100000
$paperBytes = (Get-Item $paper).Length
if ($paperBytes -lt $minBytes) {
  throw "ABORT: paper_trader.py size ($paperBytes) < $minBytes bytes. Restore from backups first (.\backups\paper_trader_pre_*.py)."
}

$bkPaper = Backup-File $paper $BACKUPS
$bkIb    = Backup-File $ibmod $BACKUPS
Write-Host "[OK] Backed up paper_trader.py -> $bkPaper"
Write-Host "[OK] Backed up pt\ib_connect.py -> $bkIb"

# --- Inject or replace IB bootstrap section in pt\ib_connect.py ---
$begin = "# === IB_BOOTSTRAP_EXTRACT v1 BEGIN ==="
$end   = "# === IB_BOOTSTRAP_EXTRACT v1 END ==="

$inject = @'
# === IB_BOOTSTRAP_EXTRACT v1 BEGIN ===
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any

# NOTE:
# This section is intentionally self-contained and safe to inject into an existing pt.ib_connect module.
# It avoids importing project-specific modules other than optional logger usage to prevent circular imports.

@dataclass(frozen=True)
class IBConnSpec:
    host: str = "127.0.0.1"
    port: int = 4002
    client_id: int = 1111
    readonly: bool = True
    timeout: float = 6.0
    account: Optional[str] = None  # optional account selection

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
    """
    Create an IBConnSpec from a parsed args namespace (argparse-like).

    Designed to work with common paper_trader args fields:
      args.host, args.port, args.clientId (or args.client_id), args.readonly (optional), args.timeout (optional)
    """
    d = defaults or IBConnSpec()
    host = getattr(args, "host", d.host) or d.host
    port = _coerce_int(getattr(args, "port", d.port), d.port)
    client_id = _coerce_int(getattr(args, "clientId", None) or getattr(args, "client_id", d.client_id), d.client_id)
    readonly = bool(getattr(args, "readonly", d.readonly))
    timeout = _coerce_float(getattr(args, "timeout", d.timeout), d.timeout)
    account = getattr(args, "account", None) or getattr(args, "ib_account", None) or d.account
    return IBConnSpec(host=host, port=port, client_id=client_id, readonly=readonly, timeout=timeout, account=account)

def bootstrap_ib(
    spec: IBConnSpec,
    *,
    logger: Optional[Any] = None,
    connect_fn: Optional[Any] = None,
) -> Any:
    """
    Centralized IB bootstrap used by paper_trader orchestrator.

    - spec: connection spec
    - logger: optional object with .info/.warning/.error
    - connect_fn: optional override for dependency injection/testing

    Returns:
      ib object (usually ib_insync.IB instance) that is connected.
    """
    # Lazy import to avoid import-time side effects/circular imports
    try:
        from ib_insync import IB  # type: ignore
    except Exception as e:
        raise RuntimeError("ib_insync is required for IB bootstrap but could not be imported") from e

    ib = IB() if connect_fn is None else connect_fn()

    if logger:
        logger.info(f"[IB] Connecting host={spec.host} port={spec.port} clientId={spec.client_id} readonly={spec.readonly} timeout={spec.timeout}")

    # ib_insync: IB.connect(host, port, clientId=..., readonly=..., timeout=...)
    try:
        ib.connect(spec.host, spec.port, clientId=spec.client_id, readonly=spec.readonly, timeout=spec.timeout)
    except TypeError:
        # fallback for older signatures
        ib.connect(spec.host, spec.port, clientId=spec.client_id, readonly=spec.readonly)

    # Optional account selection (safe no-op if not applicable)
    if spec.account and logger:
        logger.info(f"[IB] Account preference supplied: {spec.account}")

    if logger:
        try:
            ok = bool(getattr(ib, "isConnected")() if callable(getattr(ib, "isConnected", None)) else getattr(ib, "isConnected", False))
            logger.info(f"[IB] Connected={ok}")
        except Exception:
            logger.info("[IB] Connected (status unknown)")

    return ib

def bootstrap_ib_from_args(args: Any, *, logger: Optional[Any] = None) -> Any:
    """
    Convenience wrapper: builds IBConnSpec from args and connects.
    """
    spec = spec_from_args(args)
    return bootstrap_ib(spec, logger=logger)

# === IB_BOOTSTRAP_EXTRACT v1 END ===

'@

$ibText = Get-Content $ibmod -Raw -Encoding UTF8

if ($ibText -match [regex]::Escape($begin) -and $ibText -match [regex]::Escape($end)) {
  $pattern = [regex]::Escape($begin) + ".*?" + [regex]::Escape($end)
  $ibNew = [regex]::Replace($ibText, $pattern, $inject, "Singleline")
  Write-Host "[OK] Replaced existing IB bootstrap block in pt\ib_connect.py"
} else {
  if (-not $ibText.EndsWith("`n")) { $ibText += "`n" }
  $ibNew = $ibText + "`n" + $inject + "`n"
  Write-Host "[OK] Appended IB bootstrap block to pt\ib_connect.py"
}

Write-AtomicText $ibmod $ibNew

# --- Best-effort paper_trader.py patch: route connect to bootstrap_ib_from_args ---
$paperText = Get-Content $paper -Raw -Encoding UTF8
$changed = $false

# Ensure import exists (best-effort)
if ($paperText -notmatch "(?m)^\s*from\s+pt\.ib_connect\s+import\s+bootstrap_ib_from_args") {
  $head = $paperText.Substring(0, [Math]::Min(4000, $paperText.Length))
  $m = [regex]::Match($head, "(?ms)^(.*?)(\r?\n\r?\n)", "Singleline")
  if ($m.Success) {
    $insertAt = $m.Groups[1].Length
    $paperText = $paperText.Insert($insertAt, "`r`nfrom pt.ib_connect import bootstrap_ib_from_args`r`n")
  } else {
    $paperText = "from pt.ib_connect import bootstrap_ib_from_args`r`n" + $paperText
  }
  $changed = $true
  Write-Host "[OK] Added import: from pt.ib_connect import bootstrap_ib_from_args"
}

# Pattern A: ib = IB(); ib.connect(...)
$patA = "(?ms)^\s*ib\s*=\s*IB\(\)\s*\r?\n\s*ib\.connect\([^\)]*\)\s*"
if ($paperText -match $patA) {
  $paperText = [regex]::Replace($paperText, $patA, "ib = bootstrap_ib_from_args(args, logger=log)`r`n", 1)
  $changed = $true
  Write-Host "[OK] Rewired IB.connect() block -> bootstrap_ib_from_args(args, logger=log)"
}

# Pattern B: ib.connect(...) on its own line
$patB = "(?m)^\s*ib\.connect\([^\)]*\)\s*$"
if (-not ($paperText -match $patA) -and ($paperText -match $patB)) {
  $paperText = [regex]::Replace($paperText, $patB, "ib = bootstrap_ib_from_args(args, logger=log)", 1)
  $changed = $true
  Write-Host "[OK] Rewired ib.connect(...) line -> ib = bootstrap_ib_from_args(args, logger=log)"
}

if (-not $changed) {
  Write-Host "[WARN] No safe IB connect pattern matched in paper_trader.py. pt\ib_connect.py was updated, but paper_trader.py was left unchanged."
  Write-Host "       Search manually for 'IB()' / 'ib.connect(' and swap to: ib = bootstrap_ib_from_args(args, logger=log)"
} else {
  Write-AtomicText $paper $paperText
  Write-Host "[OK] paper_trader.py updated."
}

Write-Host ""
Write-Host "Verify compile:"
Write-Host "  .\.venv\Scripts\python.exe -m py_compile .\paper_trader.py .\pt\ib_connect.py"
Write-Host "Then run:"
Write-Host "  .\.venv\Scripts\python.exe -u .\paper_trader.py --host 127.0.0.1 --port 4002 --clientId 1111"
