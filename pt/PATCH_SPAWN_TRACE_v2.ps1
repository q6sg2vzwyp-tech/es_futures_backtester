# PATCH_SPAWN_TRACE_v2.ps1
# Safer spawn trace patch:
# - NEVER prepends at file top if anchor missing (aborts instead)
# - Verifies file size before/after write to prevent truncation
#
# Usage:
#   cd C:\Users\owner\Desktop\es_futures_backtester
#   Set-ExecutionPolicy -Scope Process Bypass -Force
#   .\pt\PATCH_SPAWN_TRACE_v2.ps1

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ROOT   = (Resolve-Path ".").Path
$TARGET = Join-Path $ROOT "paper_trader.py"
if (!(Test-Path $TARGET)) { throw "Could not find paper_trader.py at: $TARGET" }

function Get-Size($p) { (Get-Item -LiteralPath $p).Length }

$size0 = Get-Size $TARGET
if ($size0 -lt 50000) {
  throw "Refusing to patch: paper_trader.py looks too small already (Length=$size0). Restore from backups first."
}

# Install module if needed
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$SRC_MOD = Join-Path $SCRIPT_DIR "spawn_trace.py"
$DST_MOD_DIR = Join-Path $ROOT "pt"
$DST_MOD = Join-Path $DST_MOD_DIR "spawn_trace.py"

if (!(Test-Path $SRC_MOD)) { throw "Missing spawn_trace.py next to script: $SRC_MOD" }
New-Item -ItemType Directory -Force -Path $DST_MOD_DIR | Out-Null

$srcAbs = (Resolve-Path $SRC_MOD).Path
$dstAbs = $null
if (Test-Path $DST_MOD) { $dstAbs = (Resolve-Path $DST_MOD).Path }

if ($dstAbs -and ($srcAbs -eq $dstAbs)) {
  Write-Host "[OK] spawn_trace.py already in place: $DST_MOD"
} else {
  Copy-Item -Force $SRC_MOD $DST_MOD
  Write-Host "[OK] Installed: $DST_MOD"
}

# Backup
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$BKDIR = Join-Path $ROOT "backups"
New-Item -ItemType Directory -Force -Path $BKDIR | Out-Null
$BK = Join-Path $BKDIR ("paper_trader_pre_spawn_trace_{0}.py" -f $stamp)
Copy-Item -Force $TARGET $BK
Write-Host "[OK] Backup created: $BK"

$txt = Get-Content -LiteralPath $TARGET -Raw -Encoding UTF8

$importLine = "from pt.spawn_trace import _spawn_trace_write, _spawn_trace_init"

# If already present, do not insert again
if ($txt -match [regex]::Escape($importLine)) {
  Write-Host "[OK] spawn_trace import already present."
} else {
  # Preferred anchor: pt modules marker (same one used by io_utils patch)
  $marker = "# --- pt modules (NO side effects; import only after singleton lock acquired) ---"
  if ($txt -match [regex]::Escape($marker)) {
    $txt = $txt -replace [regex]::Escape($marker), ($marker + "`r`n" + $importLine)
    Write-Host "[OK] Inserted spawn_trace import after pt modules marker."
  } else {
    # Fallback anchor: insert after the last contiguous 'from pt.' import line in the first import block
    $m = [regex]::Match($txt, "(?ms)^(from\s+pt\.[^\r\n]+\r?\n)+(?!from\s+pt\.)")
    if ($m.Success) {
      $insertAt = $m.Index + $m.Length
      $txt = $txt.Substring(0, $insertAt) + $importLine + "`r`n" + $txt.Substring($insertAt)
      Write-Host "[OK] Inserted spawn_trace import after existing pt.* imports block."
    } else {
      throw "Could not find a safe anchor to insert spawn_trace import (no pt modules marker and no pt.* import block). Aborting."
    }
  }
}

# Remove duplicate defs if present (optional)
function Remove-DefBlock([string]$defName) {
  $pat = [regex]("(?ms)^def\s+" + [regex]::Escape($defName) + "\s*\(.*?\):\s*\r?\n(?:^[ \t].*\r?\n)+")
  if ($pat.IsMatch($script:txt)) {
    $script:txt = $pat.Replace($script:txt, "", 1)
    Write-Host "[OK] Removed in-file def $defName()."
  } else {
    Write-Host "[INFO] No in-file def $defName() found."
  }
}

Remove-DefBlock "_spawn_trace_write"
Remove-DefBlock "_spawn_trace_init"

# Write to a temp file first, then replace, then size-check
$tmpOut = $TARGET + ".tmp_patch"
Set-Content -LiteralPath $tmpOut -Value $txt -Encoding UTF8

$size1 = Get-Size $tmpOut
if ($size1 -lt ($size0 * 0.90)) {
  Remove-Item -Force $tmpOut -ErrorAction SilentlyContinue
  throw "Safety abort: patched output looks too small (before=$size0 after=$size1). NOT overwriting paper_trader.py."
}

Move-Item -Force $tmpOut $TARGET
Write-Host "[OK] Patched: $TARGET"
Write-Host "[NEXT] Compile:"
Write-Host "       .\.venv\Scripts\python.exe -m py_compile .\pt\spawn_trace.py"
Write-Host "       .\.venv\Scripts\python.exe -m py_compile .\paper_trader.py"
