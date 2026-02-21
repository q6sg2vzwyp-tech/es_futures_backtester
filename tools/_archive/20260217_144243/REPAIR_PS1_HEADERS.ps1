param(
  [string]$ToolsDir = (Join-Path (Split-Path -Parent $PSScriptRoot) "tools")
)

$ErrorActionPreference = "Continue"

if (!(Test-Path $ToolsDir)) {
  Write-Host "[WARN] ToolsDir not found: $ToolsDir" -ForegroundColor Yellow
  exit 0
}

$ps1s = Get-ChildItem -Path $ToolsDir -Filter "*.ps1" -File -ErrorAction SilentlyContinue
if (!$ps1s -or $ps1s.Count -eq 0) {
  Write-Host "[WARN] No .ps1 files found in $ToolsDir" -ForegroundColor Yellow
  exit 0
}

foreach ($f in $ps1s) {
  try {
    $raw = Get-Content $f.FullName -Raw -Encoding UTF8
    # If file starts with "\" (optionally preceded by BOM), remove that first line.
    # Also remove any leading blank lines before param if they contain "\".
    $lines = $raw -split "`r?`n", 0, "RegexMatch"
    if ($lines.Length -gt 0 -and ($lines[0].Trim() -eq "\")) {
      $bak = ($f.FullName + ".bak_hdr_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
      Copy-Item $f.FullName $bak -Force
      $fixed = ($lines[1..($lines.Length-1)] -join "`r`n")
      Set-Content -Path $f.FullName -Value $fixed -Encoding UTF8
      Write-Host "[FIX] Removed leading '\' line: $($f.Name) (backup: $([IO.Path]::GetFileName($bak)))" -ForegroundColor Green
      continue
    }

    # If first non-empty, non-comment line is not 'param(', warn.
    $first = $null
    foreach ($ln in $lines) {
      $t = $ln.Trim()
      if ($t -eq "") { continue }
      if ($t.StartsWith("#")) { continue }
      $first = $t
      break
    }
    if ($first -and -not ($first -match "^(?i)param\s*\(")) {
      Write-Host "[WARN] $($f.Name): first statement is not param(...): $first" -ForegroundColor Yellow
    }
  } catch {
    Write-Host "[WARN] Could not inspect/fix $($f.Name): $($_.Exception.Message)" -ForegroundColor Yellow
  }
}

Write-Host "[DONE] Header repair pass complete." -ForegroundColor Cyan

