# PATCH_PT_CHILD_DENY_GUARD_SAFE_v1.ps1
# Inserts a small Python guard into paper_trader.py to prevent child instances
# when the parent process command line already contains paper_trader.py.
# Idempotent (won't re-insert if marker exists).
# Safety: refuses to run if paper_trader.py is unexpectedly small.

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Info($m){ Write-Host $m }

$ROOT = (Resolve-Path ".").Path
$pt = Join-Path $ROOT "paper_trader.py"
if (!(Test-Path $pt)) { throw "paper_trader.py not found at: $pt" }

$paperBytes = (Get-Item $pt).Length
$minBytes = 100000
if ($paperBytes -lt $minBytes) {
  throw "ABORT: paper_trader.py size ($paperBytes) < $minBytes bytes. Restore from backups first (.\backups\paper_trader_pre_*.py)."
}

# backup current
$bkDir = Join-Path $ROOT "backups"
New-Item -ItemType Directory -Path $bkDir -Force | Out-Null
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$bk = Join-Path $bkDir ("paper_trader.py_" + $ts + ".bak")
Copy-Item $pt $bk -Force
Info "[OK] Backed up paper_trader.py -> $bk"

$txt = Get-Content $pt -Raw -Encoding UTF8

$marker = "PT_CHILD_DENY_GUARD v1 BEGIN"
if ($txt -match [regex]::Escape($marker)) {
  Info "[OK] Guard already present. No changes."
  exit 0
}

$block = @"
# ---- PT_CHILD_DENY_GUARD v1 BEGIN ----
# Prevents paper_trader.py from running as a child of another paper_trader.py process.
# Override by setting environment variable PT_ALLOW_CHILD=1
try:
    import os as _os_cd, sys as _sys_cd
    if _os_cd.environ.get("PT_ALLOW_CHILD","").strip().lower() not in ("1","true","yes","on"):
        try:
            import psutil as _psutil_cd
            _p_cd = _psutil_cd.Process(_os_cd.getpid())
            _parent_cd = _p_cd.parent()
            if _parent_cd is not None:
                _pcmd_cd = " ".join(_parent_cd.cmdline()).lower()
                if "paper_trader.py" in _pcmd_cd:
                    try:
                        print(f"[CHILD_DENY] parent appears to be paper_trader (ppid={_parent_cd.pid}); exiting.")
                    except Exception:
                        pass
                    _sys_cd.exit(0)
        except Exception:
            pass
except Exception:
    pass
# ---- PT_CHILD_DENY_GUARD v1 END ----
"@

# Prefer inserting right after the spawn tracer end marker if present
$anchor = "# ================== END SPAWN TRACER =================="
if ($txt -match [regex]::Escape($anchor)) {
  $new = $txt -replace ([regex]::Escape($anchor)), ($anchor + "`r`n`r`n" + $block + "`r`n")
} else {
  # fallback: insert after first line
  $lines = $txt -split "`r?`n", 2
  if ($lines.Count -ge 2) {
    $new = $lines[0] + "`r`n" + $block + "`r`n" + $lines[1]
  } else {
    $new = $txt + "`r`n" + $block + "`r`n"
  }
}

# write atomically using tools\safe_write.ps1 if present
$safeWrite = Join-Path $ROOT "tools\safe_write.ps1"
$tmp = $pt + ".tmp"
if (Test-Path $safeWrite) {
  Set-Content -Path $tmp -Value $new -Encoding UTF8
  powershell -NoProfile -ExecutionPolicy Bypass -File $safeWrite -Target $pt -Temp $tmp | Out-Null
  Remove-Item $tmp -Force -ErrorAction SilentlyContinue
} else {
  Set-Content -Path $pt -Value $new -Encoding UTF8
}

Info "[OK] Inserted PT_CHILD_DENY_GUARD into paper_trader.py"
Info ""
Info "Verify compile:"
Info "  .\.venv\Scripts\python.exe -m py_compile .\paper_trader.py"
Info "Run:"
Info "  .\.venv\Scripts\python.exe -u .\paper_trader.py --host 127.0.0.1 --port 4002 --clientId 1111"
