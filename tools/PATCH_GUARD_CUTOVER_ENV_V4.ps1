$ErrorActionPreference = "Stop"

$paper = ".\paper_trader.py"
if (-not (Test-Path $paper)) { throw "Missing: $paper" }

$bak = "$paper.bak_guard_cutover_v4_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
Copy-Item $paper $bak -Force
Write-Host "[BACKUP] $bak"

$src = Get-Content $paper -Raw -Encoding UTF8

$beginPat = '(?m)^\s*#\s*=+\s*PT_CUTOVER_TO_PT_LOOP v4\s*=+\s*$'
$endPat   = '(?m)^\s*#\s*=+\s*END PT_CUTOVER_TO_PT_LOOP v4\s*=+\s*$'

$mb = [regex]::Match($src, $beginPat)
$me = [regex]::Match($src, $endPat)

if (-not $mb.Success -or -not $me.Success -or $me.Index -le $mb.Index) {
  throw "Could not find v4 cutover begin/end markers."
}

$block = $src.Substring($mb.Index, ($me.Index + $me.Length) - $mb.Index)

# If already guarded inside the block, do nothing
if ($block -match '(?m)^\s*if\s+PT_ENABLE_CUTOVER\s*:' -or $block -match '_PT_ENABLE_CUTOVER') {
  Write-Host "[OK] v4 cutover already guarded (no changes)." -ForegroundColor Green
  exit 0
}

# Determine indentation of the marker line
$indent = ""
$mIndent = [regex]::Match($block, '(?m)^(?<i>\s*)#\s*=+\s*PT_CUTOVER_TO_PT_LOOP v4')
if ($mIndent.Success) { $indent = $mIndent.Groups["i"].Value }

# Indent the existing block by 4 spaces under the new if:
$indentedBlock = ($block -split "`r?`n" | ForEach-Object { "${indent}    $_" }) -join "`r`n"

$wrapped = @"
${indent}import os as _os
${indent}_PT_ENABLE_CUTOVER = _os.environ.get("PT_ENABLE_CUTOVER","0").strip().lower() in ("1","true","yes","on")
${indent}if _PT_ENABLE_CUTOVER:
$indentedBlock
${indent}else:
${indent}    # cutover disabled -> legacy loop below
${indent}    pass
"@

$src2 = $src.Substring(0,$mb.Index) + $wrapped + $src.Substring($me.Index + $me.Length)

Set-Content -Path $paper -Value $src2 -Encoding UTF8
Write-Host "[OK] Wrapped v4 cutover block behind PT_ENABLE_CUTOVER env guard." -ForegroundColor Green

python -m py_compile .\paper_trader.py
if ($LASTEXITCODE -ne 0) { throw "Compile failed." }
Write-Host "[OK] Compile: paper_trader.py" -ForegroundColor Green
