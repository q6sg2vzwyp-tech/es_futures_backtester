param()

$ErrorActionPreference="Stop"
$ProgressPreference="SilentlyContinue"

$root = (Get-Location).Path
$ts   = Get-Date -Format "yyyyMMdd_HHmmss"

$outDir = Join-Path $root "release"
$stage  = Join-Path $outDir ("stage_$ts")

New-Item -ItemType Directory -Force -Path $outDir | Out-Null
New-Item -ItemType Directory -Force -Path $stage  | Out-Null

$excludeDirs = @(
  '\.venv\','\logs\','\run\','\results\','\__pycache__\',
  '\.git\','\release\',
  '\_archive_broken_IGNORE\','\archive\','\pt_v7_update\','\_pt_v5_tmp\'
)

function Is-ExcludedPath([string]$rel){
  $norm = "\" + ($rel -replace '/','\') + "\"
  foreach($d in $excludeDirs){
    if($norm -like "*$d*"){ return $true }
  }
  return $false
}

Write-Host "[INFO] Staging from: $root"
Write-Host "[INFO] Stage into : $stage"

Get-ChildItem -Path $root -Recurse -File | ForEach-Object {
  $full = $_.FullName
  $rel  = $full.Substring($root.Length).TrimStart('\','/')

  if(Is-ExcludedPath $rel){ return }
  if($rel -like "*.pyc"){ return }

  $dest = Join-Path $stage $rel
  New-Item -ItemType Directory -Force -Path (Split-Path $dest -Parent) | Out-Null
  Copy-Item -Force -Path $full -Destination $dest
}

Push-Location $stage
Write-Host "[SMOKE] python:" (Get-Command python -ErrorAction SilentlyContinue).Source
python -c "import paper_trader,loop_core,pt_loop_exec; print('IMPORTS_OK')"
python -c "from strategy_core import build_signal_and_bands; print(f'STRATEGY_OK {callable(build_signal_and_bands)}')"
Pop-Location
Write-Host "[SMOKE] OK"

$zipPath = Join-Path $outDir ("es_futures_backtester_release_$ts.zip")
if(Test-Path $zipPath){ Remove-Item -Force $zipPath }
Compress-Archive -Path (Join-Path $stage "*") -DestinationPath $zipPath -Force

Write-Host "[OK] ZIP  :" $zipPath
Write-Host "[OK] STAGE:" $stage
