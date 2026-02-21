# tools/PATCH_GUARD_CUTOVER_ENV.ps1
# Wrap the PT_CUTOVER_TO_PT_LOOP block so it only runs when PT_ENABLE_CUTOVER=1/true/yes/on.
# This stops "missing symbol" churn and preserves legacy loop stability.
param(
  [string]$Paper = ".\paper_trader.py"
)

function EnvIsOn([string]$v) {
  if ($null -eq $v) { return $false }
  $s = $v.ToString().Trim().ToLower()
  return @("1","true","yes","on") -contains $s
}

if (-not (Test-Path $Paper)) { throw "Missing $Paper" }

$bak = "$Paper.bak_guard_cutover_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
Copy-Item $Paper $bak -Force
Write-Host "[BACKUP] $bak"

$src = Get-Content $Paper -Raw -Encoding UTF8

$beginPat = '(?m)^\s*#\s*=+\s*PT_CUTOVER_TO_PT_LOOP\s+v3\s*=+\s*$'
$endPat   = '(?m)^\s*#\s*=+\s*END\s+PT_CUTOVER_TO_PT_LOOP\s+v3\s*=+\s*$'

$mBegin = [regex]::Match($src, $beginPat)
if (-not $mBegin.Success) { throw "Could not find begin marker for v3 cutover." }

$mEnd = [regex]::Match($src, $endPat)
if (-not $mEnd.Success) { throw "Could not find end marker for v3 cutover." }

if ($mEnd.Index -le $mBegin.Index) { throw "End marker occurs before begin marker (unexpected)." }

# Detect already-guarded
$already = [regex]::Match($src.Substring($mBegin.Index, [Math]::Min(400, $src.Length - $mBegin.Index)), '(?m)^\s*if\s+_os\.environ\.get\("PT_ENABLE_CUTOVER"')
if ($already.Success) {
  Write-Host "[OK] Cutover block already guarded. No changes." -ForegroundColor Green
  exit 0
}

# Split to lines for stable indentation ops
$lines = $src -split "`r?`n", 0, "RegexMatch"

$beginLine = -1
$endLine = -1
for ($i=0; $i -lt $lines.Length; $i++) {
  if ($lines[$i] -match 'PT_CUTOVER_TO_PT_LOOP v3') { $beginLine = $i; break }
}
for ($j=$beginLine+1; $j -lt $lines.Length; $j++) {
  if ($lines[$j] -match 'END PT_CUTOVER_TO_PT_LOOP v3') { $endLine = $j; break }
}
if ($beginLine -lt 0 -or $endLine -lt 0) { throw "Could not locate cutover block lines." }

# Determine base indent from the marker line
$indent = ""
if ($lines[$beginLine] -match '^(?<i>\s*)#') { $indent = $Matches["i"] }

$guardLine = $indent + 'if _os.environ.get("PT_ENABLE_CUTOVER","0").strip().lower() in ("1","true","yes","on"):'
$elseLine  = $indent + 'else:'
$passLine  = $indent + '    pass  # cutover disabled -> continue legacy loop'

# Insert guard right after marker
$out = New-Object System.Collections.Generic.List[string]
for ($i=0; $i -lt $lines.Length; $i++) {
  if ($i -eq $beginLine+1) {
    $out.Add($guardLine)
  }
  # For lines between beginLine+1 .. endLine-1, indent them under the guard
  if ($i -gt $beginLine -and $i -lt $endLine) {
    $out.Add($indent + "    " + $lines[$i])
  } elseif ($i -eq $endLine) {
    # Close the if with else/pass before end marker
    $out.Add($elseLine)
    $out.Add($passLine)
    $out.Add($lines[$i])
  } else {
    $out.Add($lines[$i])
  }
}

$new = ($out -join "`r`n")

Set-Content -Path $Paper -Value $new -Encoding UTF8

# Compile check
.\.venv\Scripts\python.exe -m py_compile $Paper | Out-Host
if ($LASTEXITCODE -ne 0) { throw "Compile failed after patch." }

Write-Host "[OK] Guarded v3 cutover block behind PT_ENABLE_CUTOVER env var." -ForegroundColor Green
