param(
  [string]$RepoRoot = (Get-Location).Path
)

$ErrorActionPreference = "Continue"

function Move-IfExists($path, $destDir) {
  if (Test-Path $path) {
    New-Item -ItemType Directory -Force -Path $destDir | Out-Null
    $name = Split-Path $path -Leaf
    $dest = Join-Path $destDir $name
    Write-Host "[MOVE] $path  ->  $dest" -ForegroundColor Cyan
    try {
      Move-Item -Force -Path $path -Destination $dest
    } catch {
      Write-Host "[WARN] Failed to move $path : $($_.Exception.Message)" -ForegroundColor Yellow
    }
  }
}

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$qroot = Join-Path $RepoRoot ("quarantine\" + $ts)
New-Item -ItemType Directory -Force -Path $qroot | Out-Null

# 1) staging folders
Move-IfExists (Join-Path $RepoRoot "UPLOAD_STAGE") (Join-Path $qroot "staging")

# 2) snapshots
$runDir = Join-Path $RepoRoot "run"
if (Test-Path $runDir) {
  $snap = Get-ChildItem -Path $runDir -Force -ErrorAction SilentlyContinue | Where-Object { $_.Name -like "snapshot_*" }
  foreach ($item in $snap) {
    Move-IfExists $item.FullName (Join-Path $qroot "run_snapshots")
  }
}

# 3) patches quarantine folders under tools
$toolsDir = Join-Path $RepoRoot "tools"
if (Test-Path $toolsDir) {
  $pq = Get-ChildItem -Path $toolsDir -Force -ErrorAction SilentlyContinue | Where-Object { $_.Name -like "patches_quarantine_*" }
  foreach ($item in $pq) {
    Move-IfExists $item.FullName (Join-Path $qroot "tools_quarantine")
  }
}

# 4) misnamed *.py that look like PowerShell (move-only)
$misDir = Join-Path $qroot "misnamed_py"
$pyFiles = Get-ChildItem -Path $RepoRoot -Recurse -File -Filter "*.py" -ErrorAction SilentlyContinue |
  Where-Object { $_.FullName -notmatch "\\.venv\\" -and $_.FullName -notmatch "\\quarantine\\" }

foreach ($f in $pyFiles) {
  try {
    $head = Get-Content $f.FullName -TotalCount 5 -ErrorAction SilentlyContinue
    $headText = ($head -join "`n")
    if ($headText -match "(?i)^\s*PowerShell\s+\d" -or $headText -match "(?i)^\s*PS\s+[A-Z]:" -or $headText -match "(?i)Set-Location|ExecutionPolicy|Get-CimInstance") {
      New-Item -ItemType Directory -Force -Path $misDir | Out-Null
      $rel = $f.FullName.Substring($RepoRoot.Length).TrimStart("\")
      $destPath = Join-Path $misDir $rel
      $destFolder = Split-Path $destPath -Parent
      New-Item -ItemType Directory -Force -Path $destFolder | Out-Null
      Write-Host "[MOVE] misnamed .py -> $destPath" -ForegroundColor Cyan
      Move-Item -Force -Path $f.FullName -Destination $destPath -ErrorAction SilentlyContinue
    }
  } catch {
    # ignore
  }
}

Write-Host ""
Write-Host "[DONE] Quarantine created at: $qroot" -ForegroundColor Green
Write-Host "       (Move-only. Restore by moving items back if needed.)" -ForegroundColor Green
