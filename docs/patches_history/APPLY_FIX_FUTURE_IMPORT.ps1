# APPLY_FIX_FUTURE_IMPORT.ps1
# Ensures "from __future__ import annotations" is the first executable statement in paper_trader.py
# Creates a quarantine backup before modifying.
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ROOT = (Resolve-Path ".").Path
$target = Join-Path $ROOT "paper_trader.py"
if (-not (Test-Path $target)) { throw "paper_trader.py not found in $ROOT" }

# --- quarantine backup folder ---
$stamp = (Get-Date).ToString("yyyyMMdd_HHmmss")
$quar = Join-Path $ROOT ("tools\patches_quarantine_" + $stamp)
New-Item -ItemType Directory -Force -Path $quar | Out-Null
Copy-Item $target (Join-Path $quar "paper_trader.py.bak") -Force

# --- read file as UTF-8 (preserve) ---
$lines = Get-Content -LiteralPath $target -Encoding UTF8

# Find future import line (exact match ignoring whitespace)
$idx = -1
for ($i=0; $i -lt $lines.Count; $i++) {
  if ($lines[$i].Trim() -eq "from __future__ import annotations") { $idx = $i; break }
}
if ($idx -lt 0) {
  Write-Host "[PATCH] No future import found; nothing to do."
  exit 0
}

# Determine insertion point: keep optional shebang and encoding cookie at very top
$ins = 0
if ($lines.Count -gt 0 -and $lines[0].StartsWith("#!")) { $ins = 1 }
if ($lines.Count -gt $ins -and $lines[$ins] -match '^\s*#.*coding[:=]\s*[-\w]+' ) { $ins += 1 }

# Remove the existing future import line
$futureLine = $lines[$idx].TrimEnd()
$lines2 = @()
for ($i=0; $i -lt $lines.Count; $i++) {
  if ($i -ne $idx) { $lines2 += $lines[$i] }
}

# If the future import is already at insertion point, done
if ($ins -lt $lines2.Count -and $lines2[$ins].Trim() -eq "from __future__ import annotations") {
  Write-Host "[PATCH] Future import already at top."
  exit 0
}

# Insert future import at insertion point
$lines3 = @()
for ($i=0; $i -lt $lines2.Count; $i++) {
  if ($i -eq $ins) { $lines3 += $futureLine }
  $lines3 += $lines2[$i]
}
if ($ins -ge $lines2.Count) {
  # file shorter than insertion point
  $lines3 = $lines2 + @($futureLine)
}

# Ensure a blank line after the future import (PEP 236 style), without duplicating excessive blanks
$pos = $ins
if ($pos -ge 0 -and $pos -lt $lines3.Count) {
  $after = $pos + 1
  if ($after -lt $lines3.Count -and $lines3[$after].Trim() -ne "") {
    $lines3 = $lines3[0..$pos] + @("") + $lines3[$after..($lines3.Count-1)]
  }
}

# Write back
Set-Content -LiteralPath $target -Value $lines3 -Encoding UTF8

Write-Host "[PATCH] Moved future import to top."
Write-Host "[PATCH] Backup saved to: $quar"
Write-Host "[PATCH] Verify with:"
Write-Host "  Get-Content .\paper_trader.py -TotalCount 12"
