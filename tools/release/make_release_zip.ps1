param()

$ErrorActionPreference = "Stop"
$root   = (Get-Location).Path
$ts     = Get-Date -Format "yyyyMMdd_HHmmss"
$outDir = Join-Path $root "release"
$stage  = Join-Path $outDir ("stage_$ts")

New-Item -ItemType Directory -Force -Path $outDir | Out-Null
New-Item -ItemType Directory -Force -Path $stage  | Out-Null

$exclude = @(
  ".venv",
  "logs",
  "run",
  "results",
  "__pycache__",
  ".git",
  "release"
)

# Copy working tree into stage (excluding runtime artifacts)
Get-ChildItem -Path $root -Recurse -File | ForEach-Object {
  $full = $_.FullName
  $rel  = $full.Substring($root.Length).TrimStart('\','/')

  foreach ($d in $exclude) {
    if ($rel -like "$d\*" -or $rel -eq $d) { return }
  }

  if ($rel -like "*\__pycache__\*" -or $rel -like "*.pyc") { return }

  $dest = Join-Path $stage $rel
  New-Item -ItemType Directory -Force -Path (Split-Path $dest -Parent) | Out-Null
  Copy-Item -Force -Path $full -Destination $dest
}

# Add smoke runner into stage
$smokePath = Join-Path $stage "smoke_imports.ps1"
@"
`$ErrorActionPreference="Stop"
Write-Host "[SMOKE] python:" (Get-Command python -ErrorAction SilentlyContinue).Source
python -c "import paper_trader,loop_core,pt_loop_exec; print('IMPORTS_OK')"
python -c "from strategy_core import build_signal_and_bands; print(f'STRATEGY_OK {callable(build_signal_and_bands)}')"
Write-Host "[SMOKE] OK"
"@ | Set-Content -Encoding UTF8 -Path $smokePath

# Create zip
$zipPath = Join-Path $outDir ("es_futures_backtester_release_$ts.zip")
if (Test-Path $zipPath) { Remove-Item -Force $zipPath }
Compress-Archive -Path (Join-Path $stage "*") -DestinationPath $zipPath -Force

Write-Host "[OK] Release zip:" $zipPath
Write-Host "[OK] Smoke script:" $smokePath
