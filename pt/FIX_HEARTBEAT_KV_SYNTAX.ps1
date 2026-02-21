# FIX_HEARTBEAT_KV_SYNTAX.ps1
# Fixes the SyntaxError introduced by the HB_KV_SIDEcar patch:
#   "pass        except Exception as e:"
# Also switches _hbkv_write(p) -> _hbkv_write(payload) since your dict is named 'payload'.
#
# Usage:
#   cd C:\Users\owner\Desktop\es_futures_backtester
#   Set-ExecutionPolicy -Scope Process Bypass -Force
#   .\pt\FIX_HEARTBEAT_KV_SYNTAX.ps1

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ROOT   = (Resolve-Path ".").Path
$TARGET = Join-Path $ROOT "paper_trader.py"
if (!(Test-Path $TARGET)) { throw "Could not find paper_trader.py at: $TARGET" }

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$BKDIR = Join-Path $ROOT "backups"
New-Item -ItemType Directory -Force -Path $BKDIR | Out-Null
$BK = Join-Path $BKDIR ("paper_trader_pre_hbkv_fix_{0}.py" -f $stamp)
Copy-Item -Force $TARGET $BK
Write-Host "[OK] Backup created: $BK"

$txt = Get-Content -LiteralPath $TARGET -Raw -Encoding UTF8

# 1) Fix glued 'pass/except' token (missing newline)
$broken = "pass        except Exception as e:"
if ($txt -match [regex]::Escape($broken)) {
  $txt = $txt -replace [regex]::Escape($broken), "pass`r`n        except Exception as e:"
  Write-Host "[OK] Fixed glued 'pass/except' syntax."
} else {
  Write-Host "[INFO] Glued token not found (maybe already fixed)."
}

# 2) Correct wrong variable name
$txt2 = $txt -replace "_hbkv_write\(\s*p\s*\)", "_hbkv_write(payload)"
if ($txt2 -ne $txt) {
  Write-Host "[OK] Switched _hbkv_write(p) -> _hbkv_write(payload)."
  $txt = $txt2
}

Set-Content -LiteralPath $TARGET -Value $txt -Encoding UTF8
Write-Host "[OK] Wrote: $TARGET"
Write-Host "[NEXT] Compile check:"
Write-Host "       .\.venv\Scripts\python.exe -m py_compile .\paper_trader.py"
