# PATCH_SPAWN_TRACE_v1.ps1
# Consolidate spawn trace logic into pt\spawn_trace.py and remove duplicate defs from paper_trader.py.
#
# Usage:
#   cd C:\Users\owner\Desktop\es_futures_backtester
#   Set-ExecutionPolicy -Scope Process Bypass -Force
#   .\pt\PATCH_SPAWN_TRACE_v1.ps1
#
# Enable tracing:
#   $env:PT_SPAWN_TRACE="1"
#   $env:PT_SPAWN_TRACE_STACK="1"   # optional

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ROOT   = (Resolve-Path ".").Path
$TARGET = Join-Path $ROOT "paper_trader.py"
if (!(Test-Path $TARGET)) { throw "Could not find paper_trader.py at: $TARGET" }

# Ensure pt\spawn_trace.py exists next to this script
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

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$BKDIR = Join-Path $ROOT "backups"
New-Item -ItemType Directory -Force -Path $BKDIR | Out-Null
$BK = Join-Path $BKDIR ("paper_trader_pre_spawn_trace_{0}.py" -f $stamp)
Copy-Item -Force $TARGET $BK
Write-Host "[OK] Backup created: $BK"

$txt = Get-Content -LiteralPath $TARGET -Raw -Encoding UTF8

# 1) Ensure import line exists after pt modules marker
$importLine = "from pt.spawn_trace import _spawn_trace_write, _spawn_trace_init"
if ($txt -notmatch [regex]::Escape($importLine)) {
  $marker = "# --- pt modules (NO side effects; import only after singleton lock acquired) ---"
  if ($txt -match [regex]::Escape($marker)) {
    $txt = $txt -replace [regex]::Escape($marker), ($marker + "`r`n" + $importLine)
    Write-Host "[OK] Inserted spawn_trace import after pt modules marker."
  } else {
    $txt = $importLine + "`r`n" + $txt
    Write-Host "[WARN] pt modules marker not found; prepended spawn_trace import at file top."
  }
} else {
  Write-Host "[OK] spawn_trace import already present."
}

# 2) Remove duplicate function defs if present
function Remove-DefBlock($defName) {
  $pat = [regex]("(?ms)^def\s+" + [regex]::Escape($defName) + "\s*\(.*?\):\s*\n(?:^[ \t].*\n)+")
  if ($pat.IsMatch($txt)) {
    $script:txt = $pat.Replace($script:txt, "", 1)
    Write-Host "[OK] Removed in-file def $defName()."
  } else {
    Write-Host "[INFO] No in-file def $defName() found."
  }
}

Remove-DefBlock "_spawn_trace_write"
Remove-DefBlock "_spawn_trace_init"

Set-Content -LiteralPath $TARGET -Value $txt -Encoding UTF8
Write-Host "[OK] Patched: $TARGET"
Write-Host "[NEXT] Compile:"
Write-Host "       .\.venv\Scripts\python.exe -m py_compile .\pt\spawn_trace.py"
Write-Host "       .\.venv\Scripts\python.exe -m py_compile .\paper_trader.py"
