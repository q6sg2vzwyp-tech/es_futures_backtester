Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ROOT = if ($PSScriptRoot) { (Resolve-Path (Join-Path $PSScriptRoot "..")).Path } else { (Resolve-Path ".").Path }
$STAMP = Get-Date -Format "yyyyMMdd_HHmmss"

# Backup existing files we are going to touch
$BK = Join-Path $ROOT ("tools\patches_quarantine_" + $STAMP)
New-Item -ItemType Directory -Force -Path $BK | Out-Null

function Backup-File([string]$relPath) {
  $src = Join-Path $ROOT $relPath
  if (Test-Path $src) {
    $dst = Join-Path $BK $relPath
    New-Item -ItemType Directory -Force -Path (Split-Path $dst -Parent) | Out-Null
    Copy-Item -Force $src $dst
  }
}

Backup-File "paper_trader.py"
Backup-File "pt\singleton.py"
Backup-File "pt\utils_time.py"
Backup-File "pt\args.py"
Backup-File "pt\ib_connect.py"
Backup-File "pt\__init__.py"

# Apply patch (assumes you copied patch files into ROOT next to this script or into a temp folder)
# If you're running this from inside the extracted zip, $PATCH points there:
$PATCH = $PSScriptRoot

# Ensure pt package exists
New-Item -ItemType Directory -Force -Path (Join-Path $ROOT "pt") | Out-Null

Copy-Item -Force (Join-Path $PATCH "paper_trader.py") (Join-Path $ROOT "paper_trader.py")
Copy-Item -Force (Join-Path $PATCH "pt\__init__.py") (Join-Path $ROOT "pt\__init__.py")
Copy-Item -Force (Join-Path $PATCH "pt\singleton.py") (Join-Path $ROOT "pt\singleton.py")
Copy-Item -Force (Join-Path $PATCH "pt\utils_time.py") (Join-Path $ROOT "pt\utils_time.py")
Copy-Item -Force (Join-Path $PATCH "pt\args.py") (Join-Path $ROOT "pt\args.py")
Copy-Item -Force (Join-Path $PATCH "pt\ib_connect.py") (Join-Path $ROOT "pt\ib_connect.py")

Write-Host ""
Write-Host "Applied Step1-4 patch."
Write-Host "Backup stored at: $BK"

# Quick compile check (optional)
$PY = Join-Path $ROOT ".venv\Scripts\python.exe"
if (Test-Path $PY) {
  & $PY -c "import py_compile; py_compile.compile('paper_trader.py', doraise=True); py_compile.compile(r'pt\singleton.py', doraise=True); py_compile.compile(r'pt\utils_time.py', doraise=True); py_compile.compile(r'pt\args.py', doraise=True); py_compile.compile(r'pt\ib_connect.py', doraise=True); print('compile OK')"
} else {
  Write-Host "NOTE: .venv python not found at $PY. Skipping compile check."
}
