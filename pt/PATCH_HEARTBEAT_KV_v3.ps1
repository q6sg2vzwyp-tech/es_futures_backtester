# PATCH_HEARTBEAT_KV_v3.ps1
# Patches paper_trader.py to write a key=value heartbeat sidecar alongside heartbeat.txt.
# This version patches the HEARTBEAT FILE WRITER (where heartbeat.txt is written),
# so it does not depend on hb_update() internals.
#
# Usage:
#   cd C:\Users\owner\Desktop\es_futures_backtester
#   Set-ExecutionPolicy -Scope Process Bypass -Force
#   .\pt\PATCH_HEARTBEAT_KV_v3.ps1

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ROOT   = (Resolve-Path ".").Path
$TARGET = Join-Path $ROOT "paper_trader.py"
if (!(Test-Path $TARGET)) { throw "Could not find paper_trader.py at: $TARGET" }

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$BKDIR = Join-Path $ROOT "backups"
New-Item -ItemType Directory -Force -Path $BKDIR | Out-Null
$BK = Join-Path $BKDIR ("paper_trader_pre_hbkv_{0}.py" -f $stamp)
Copy-Item -Force $TARGET $BK
Write-Host "[OK] Backup created: $BK"

$txt = Get-Content -LiteralPath $TARGET -Raw -Encoding UTF8

if ($txt -match "HB_KV_SIDEcar") {
  Write-Host "[SKIP] HB KV patch already present."
  exit 0
}

$helper = @'
# === HB_KV_SIDEcar (auto-patched) ============================================
import os as _os_hbkv

_HB_KV_PATH = r".\run\heartbeat_kv.txt"

def _hbkv_write_atomic(path: str, content: str) -> None:
    try:
        d = _os_hbkv.path.dirname(path)
        if d:
            _os_hbkv.makedirs(d, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(content)
        _os_hbkv.replace(tmp, path)
    except Exception:
        return

def _hbkv_write(payload: dict) -> None:
    try:
        lines = [f"{k}={v}" for k, v in payload.items()]
        _hbkv_write_atomic(_HB_KV_PATH, "\n".join(lines) + "\n")
    except Exception:
        return
# ============================================================================
'@

# Patch point: the heartbeat writer line uses a dict named 'p' and calls replace(tmp, hb_path)
$rx = [regex]::new("(?m)^(?<indent>\s*)(?<mod>[A-Za-z_][A-Za-z0-9_]*)\.replace\(\s*tmp\s*,\s*hb_path\s*\)\s*$")
$m = $rx.Match($txt)
if (-not $m.Success) {
  throw "Could not locate the heartbeat writer line: *.replace(tmp, hb_path). Aborting."
}

# Insert helper near the start (after first blank line)
$insertPos = $txt.IndexOf("`n`n")
if ($insertPos -lt 0) { $insertPos = 0 }
$txt = $txt.Substring(0, $insertPos + 2) + $helper + "`r`n" + $txt.Substring($insertPos + 2)

# Re-find the replace line (first occurrence) and append KV write right after it
$m2 = $rx.Match($txt)
if (-not $m2.Success) {
  throw "Internal error: could not re-find replace(tmp, hb_path) after insertion."
}
$indent = $m2.Groups["indent"].Value

$append = @"
${indent}try:
${indent}    _hbkv_write(p)
${indent}except Exception:
${indent}    pass
"@

$lineEnd = $txt.IndexOf("`n", $m2.Index)
if ($lineEnd -lt 0) { $lineEnd = $txt.Length - 1 }
$txt = $txt.Substring(0, $lineEnd + 1) + $append + $txt.Substring($lineEnd + 1)

Set-Content -LiteralPath $TARGET -Value $txt -Encoding UTF8
Write-Host "[OK] Patched: $TARGET"
Write-Host "[OK] KV heartbeat will be written to: .\run\heartbeat_kv.txt"
