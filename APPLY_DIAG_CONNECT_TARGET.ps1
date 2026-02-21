Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ROOT = (Resolve-Path ".").Path
$PT = Join-Path $ROOT "paper_trader.py"
if (-not (Test-Path $PT)) { throw "paper_trader.py not found in $ROOT" }

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$QDIR = Join-Path $ROOT ("tools\patches_quarantine_{0}" -f $ts)
New-Item -ItemType Directory -Force -Path $QDIR | Out-Null
Copy-Item $PT (Join-Path $QDIR "paper_trader.py.BEFORE_DIAG") -Force

$lines = Get-Content $PT -Raw

# Insert diagnostic right before: cid = connect_with_retries()
$pattern1 = '(?m)^(?<indent>\s*)cid\s*=\s*connect_with_retries\(\)\s*$'
if ($lines -match $pattern1) {
  $lines = [regex]::Replace($lines, $pattern1, {
    param($m)
    $ind = $m.Groups["indent"].Value
    $diag = @(
      "$ind" + "try:",
      "$ind" + "    log({""evt"":""diag_connect_target"",""host"":getattr(args,'host',None),""port"":getattr(args,'port',None),""clientId"":getattr(args,'clientId',None),""timeout"":getattr(args,'connect_timeout_sec',None)})",
      "$ind" + "except Exception:",
      "$ind" + "    pass"
    ) -join "`n"
    return $diag + "`n" + $m.Value
  }, 1)
} else {
  # Alternative older signature
  $pattern1b = '(?m)^(?<indent>\s*)cid\s*=\s*connect_with_retries\([^\)]*\)\s*$'
  if ($lines -match $pattern1b) {
    $lines = [regex]::Replace($lines, $pattern1b, {
      param($m)
      $ind = $m.Groups["indent"].Value
      $diag = @(
        "$ind" + "try:",
        "$ind" + "    log({""evt"":""diag_connect_target"",""host"":getattr(args,'host',None),""port"":getattr(args,'port',None),""clientId"":getattr(args,'clientId',None),""timeout"":getattr(args,'connect_timeout_sec',None)})",
        "$ind" + "except Exception:",
        "$ind" + "    pass"
      ) -join "`n"
      return $diag + "`n" + $m.Value
    }, 1)
  } else {
    throw "Could not find connect_with_retries() call site in paper_trader.py. Search for 'connect_with_retries' and patch manually."
  }
}

# Insert diagnostic inside connect_with_retries() right before ib.connect(...)
$pattern2 = '(?m)^(?<indent>\s*)ib\.connect\(\s*args\.host\s*,\s*args\.port\s*,\s*clientId\s*=\s*cid\s*,\s*timeout\s*=\s*args\.connect_timeout_sec\s*\)\s*$'
if ($lines -match $pattern2) {
  $lines = [regex]::Replace($lines, $pattern2, {
    param($m)
    $ind = $m.Groups["indent"].Value
    $diag = @(
      "$ind" + "try:",
      "$ind" + "    log({""evt"":""diag_ib_connect_call"",""host"":args.host,""port"":args.port,""cid"":cid,""timeout"":args.connect_timeout_sec})",
      "$ind" + "except Exception:",
      "$ind" + "    pass"
    ) -join "`n"
    return $diag + "`n" + $m.Value
  }, 1)
} else {
  $pattern2b = '(?m)^(?<indent>\s*)ib\.connect\(\s*args\.host\s*,\s*args\.port\s*,\s*clientId\s*=\s*cid\s*,\s*timeout\s*=\s*[^\)]*\)\s*$'
  if ($lines -match $pattern2b) {
    $lines = [regex]::Replace($lines, $pattern2b, {
      param($m)
      $ind = $m.Groups["indent"].Value
      $diag = @(
        "$ind" + "try:",
        "$ind" + "    log({""evt"":""diag_ib_connect_call"",""host"":args.host,""port"":args.port,""cid"":cid})",
        "$ind" + "except Exception:",
        "$ind" + "    pass"
      ) -join "`n"
      return $diag + "`n" + $m.Value
    }, 1)
  } else {
    Write-Warning "Could not find exact ib.connect(args.host, args.port, clientId=cid, ...) line; skipping inner injection."
  }
}

Set-Content -Path $PT -Value $lines -Encoding utf8
Copy-Item $PT (Join-Path $QDIR "paper_trader.py.AFTER_DIAG") -Force

Write-Host "[PATCH] Added diagnostic log lines for connect target + ib.connect call."
Write-Host "[PATCH] Backup saved to: $QDIR"
Write-Host "[PATCH] Run and look for evt=diag_connect_target and evt=diag_ib_connect_call"
