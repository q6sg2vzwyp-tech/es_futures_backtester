# FIX_DIAG_CONNECT_SYNTAX.ps1
# Repairs invalid one-line try/except inserted by previous diagnostic patch.
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ROOT = (Resolve-Path ".").Path
$FILE = Join-Path $ROOT "paper_trader.py"
if (!(Test-Path $FILE)) { throw "paper_trader.py not found in $ROOT" }

# Backup
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$QDIR = Join-Path $ROOT ("tools\patches_quarantine_{0}" -f $stamp)
New-Item -ItemType Directory -Force -Path $QDIR | Out-Null
Copy-Item $FILE (Join-Path $QDIR "paper_trader.py.BEFORE_FIX_DIAG_SYNTAX") -Force

$lines = Get-Content $FILE -Encoding UTF8

function Replace-OneLineTryExcept([string[]]$inLines, [string]$needle, [string]$evtName) {
    $out = New-Object System.Collections.Generic.List[string]
    $replaced = $false
    foreach ($ln in $inLines) {
        if (-not $replaced -and $ln -like "*$needle*") {
            # Preserve indentation from the original line
            $m = [regex]::Match($ln, '^\s*')
            $indent = $m.Value
            $out.Add("${indent}try:")
            $out.Add("${indent}    log({`"evt`":`"$evtName`",`"host`":args.host,`"port`":args.port,`"clientId`":args.clientId,`"connect_timeout_sec`":args.connect_timeout_sec})")
            $out.Add("${indent}except Exception:")
            $out.Add("${indent}    pass")
            $replaced = $true
        } else {
            $out.Add($ln)
        }
    }
    return ,@($out), $replaced
}

function Replace-OneLineTryExceptCID([string[]]$inLines, [string]$needle, [string]$evtName) {
    $out = New-Object System.Collections.Generic.List[string]
    $replaced = $false
    foreach ($ln in $inLines) {
        if (-not $replaced -and $ln -like "*$needle*") {
            $m = [regex]::Match($ln, '^\s*')
            $indent = $m.Value
            $out.Add("${indent}try:")
            $out.Add("${indent}    log({`"evt`":`"$evtName`",`"host`":args.host,`"port`":args.port,`"cid`":cid,`"timeout`":args.connect_timeout_sec})")
            $out.Add("${indent}except Exception:")
            $out.Add("${indent}    pass")
            $replaced = $true
        } else {
            $out.Add($ln)
        }
    }
    return ,@($out), $replaced
}

# Fix diag_connect_target (if present as one-line try/except)
$lines2, $r1 = Replace-OneLineTryExcept $lines 'evt":"diag_connect_target"' 'diag_connect_target'

# Fix diag_ib_connect_call (if present as one-line try/except)
$lines3, $r2 = Replace-OneLineTryExceptCID $lines2 'evt":"diag_ib_connect_call"' 'diag_ib_connect_call'

# As an extra safety net: fix any "try: ... except Exception: ... pass" that contains our diag events
# (covers cases where formatting differs slightly)
$fixedAny = $false
$final = New-Object System.Collections.Generic.List[string]
foreach ($ln in $lines3) {
    if ($ln -match '^\s*try:\s*.*evt":"diag_connect_target".*except\s+Exception:\s*.*pass\s*$') {
        $indent = ([regex]::Match($ln,'^\s*')).Value
        $final.Add("${indent}try:")
        $final.Add("${indent}    log({`"evt`":`"diag_connect_target`",`"host`":args.host,`"port`":args.port,`"clientId`":args.clientId,`"connect_timeout_sec`":args.connect_timeout_sec})")
        $final.Add("${indent}except Exception:")
        $final.Add("${indent}    pass")
        $fixedAny = $true
    } elseif ($ln -match '^\s*try:\s*.*evt":"diag_ib_connect_call".*except\s+Exception:\s*.*pass\s*$') {
        $indent = ([regex]::Match($ln,'^\s*')).Value
        $final.Add("${indent}try:")
        $final.Add("${indent}    log({`"evt`":`"diag_ib_connect_call`",`"host`":args.host,`"port`":args.port,`"cid`":cid,`"timeout`":args.connect_timeout_sec})")
        $final.Add("${indent}except Exception:")
        $final.Add("${indent}    pass")
        $fixedAny = $true
    } else {
        $final.Add($ln)
    }
}

Set-Content -Path $FILE -Value $final.ToArray() -Encoding UTF8

Write-Host "[PATCH] Fixed diag syntax (connect target / ib.connect call)." -ForegroundColor Green
Write-Host ("[PATCH] Backup saved to: {0}" -f $QDIR) -ForegroundColor Yellow
Write-Host "[PATCH] Verify by running:" -ForegroundColor Yellow
Write-Host "  .\.venv\Scripts\python.exe -c `"import py_compile; py_compile.compile('paper_trader.py', doraise=True); print('paper_trader OK')`""
