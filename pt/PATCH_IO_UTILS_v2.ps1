# PATCH_IO_UTILS_v2.ps1
# Fix: do not Copy-Item when SRC_MOD == DST_MOD (same file).
#
# Usage:
#   cd C:\Users\owner\Desktop\es_futures_backtester
#   Set-ExecutionPolicy -Scope Process Bypass -Force
#   .\pt\PATCH_IO_UTILS_v2.ps1

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ROOT   = (Resolve-Path ".").Path
$TARGET = Join-Path $ROOT "paper_trader.py"
if (!(Test-Path $TARGET)) { throw "Could not find paper_trader.py at: $TARGET" }

$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$SRC_MOD = Join-Path $SCRIPT_DIR "io_utils.py"
$DST_MOD_DIR = Join-Path $ROOT "pt"
$DST_MOD = Join-Path $DST_MOD_DIR "io_utils.py"

if (!(Test-Path $SRC_MOD)) { throw "Missing io_utils.py next to script: $SRC_MOD" }
New-Item -ItemType Directory -Force -Path $DST_MOD_DIR | Out-Null

$srcAbs = (Resolve-Path $SRC_MOD).Path
$dstAbs = $null
if (Test-Path $DST_MOD) { $dstAbs = (Resolve-Path $DST_MOD).Path }

if ($dstAbs -and ($srcAbs -eq $dstAbs)) {
  Write-Host "[OK] io_utils.py already in place: $DST_MOD"
} else {
  Copy-Item -Force $SRC_MOD $DST_MOD
  Write-Host "[OK] Installed: $DST_MOD"
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$BKDIR = Join-Path $ROOT "backups"
New-Item -ItemType Directory -Force -Path $BKDIR | Out-Null
$BK = Join-Path $BKDIR ("paper_trader_pre_io_utils_{0}.py" -f $stamp)
Copy-Item -Force $TARGET $BK
Write-Host "[OK] Backup created: $BK"

$txt = Get-Content -LiteralPath $TARGET -Raw -Encoding UTF8

# 1) Insert import after pt modules marker if present
$importLine = "from pt.io_utils import mkdirs, write_text_atomic, write_json_line_atomic"
if ($txt -notmatch [regex]::Escape($importLine)) {
  $marker = "# --- pt modules (NO side effects; import only after singleton lock acquired) ---"
  if ($txt -match [regex]::Escape($marker)) {
    $txt = $txt -replace [regex]::Escape($marker), ($marker + "`r`n" + $importLine)
    Write-Host "[OK] Inserted io_utils import after pt modules marker."
  } else {
    $txt = $importLine + "`r`n" + $txt
    Write-Host "[WARN] pt modules marker not found; prepended io_utils import at file top."
  }
} else {
  Write-Host "[OK] io_utils import already present."
}

# 2) Replace def mkdirs(...) with a thin wrapper (first occurrence only)
$re = [regex]'(?ms)^def\s+mkdirs\s*\(.*?\):\s*\n(?:^[ \t].*\n)+'
if ($re.IsMatch($txt)) {
  $wrapper = @"
def mkdirs(path: str) -> None:
    # thin wrapper (moved to pt.io_utils)
    from pt.io_utils import mkdirs as _pt_mkdirs
    return _pt_mkdirs(path)

"@
  $txt = $re.Replace($txt, $wrapper, 1)
  Write-Host "[OK] Replaced mkdirs() implementation with thin wrapper."
} else {
  Write-Host "[INFO] No mkdirs() definition found in paper_trader.py (may already be extracted)."
}

# 3) Replace common atomic helper names if present
function Replace-AtomicHelper($name) {
  $pat = [regex]("(?ms)^def\s+" + [regex]::Escape($name) + "\s*\(.*?\):\s*\n(?:^[ \t].*\n)+")
  if ($pat.IsMatch($txt)) {
    $wrapper = @"
def $name(path: str, content: str) -> None:
    # thin wrapper (moved to pt.io_utils)
    from pt.io_utils import write_text_atomic as _pt_write
    return _pt_write(path, content)

"@
    $script:txt = $pat.Replace($script:txt, $wrapper, 1)
    Write-Host "[OK] Replaced $name() with thin wrapper."
  }
}

Replace-AtomicHelper "_write_atomic"
Replace-AtomicHelper "_hbkv_write_atomic"
Replace-AtomicHelper "_hb_write_atomic"

Set-Content -LiteralPath $TARGET -Value $txt -Encoding UTF8
Write-Host "[OK] Patched: $TARGET"
Write-Host "[NEXT] Compile:"
Write-Host "       .\.venv\Scripts\python.exe -m py_compile .\pt\io_utils.py"
Write-Host "       .\.venv\Scripts\python.exe -m py_compile .\paper_trader.py"
