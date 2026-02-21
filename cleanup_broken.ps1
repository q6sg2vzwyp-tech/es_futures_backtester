# cleanup_broken.ps1
# Moves known-broken / unused Python files into archive_broken

# Use the current directory as the root
$Root = Get-Location
$Quarantine = Join-Path $Root "archive_broken"

Write-Host "Base folder: $Root"
Write-Host "Quarantine folder: $Quarantine"
Write-Host ""

# Make sure the quarantine folder exists
if (-not (Test-Path $Quarantine)) {
    Write-Host "Creating quarantine folder..." -ForegroundColor Cyan
    New-Item -ItemType Directory -Path $Quarantine | Out-Null
}

# List of RELATIVE paths to broken/old files we want to move
$filesToQuarantine = @(
    "archive\maint_20251116_111141\backups\paper_trader_line2008_backup.py",
    "archive\maint_20251116_111141\backups\paper_trader_pre_fix_20251112_132424.py",
    "archive\maint_20251116_111141\backups\paper_trader_pre_fix_20251112_133529.py",
    "archive\unused\missed_reasons.py",
    "archive\unused\quick_reward_dump.py",
    "archive\unused\run_ppt_trader_supervisor.py",
    "archive\unused\twoweek_review.py",
    "core\paper_trader.py"   # old monolith, we now use root paper_trader.py
)

foreach ($relPath in $filesToQuarantine) {
    $fullPath = Join-Path $Root $relPath

    if (Test-Path $fullPath) {
        Write-Host "Moving $relPath -> archive_broken" -ForegroundColor Yellow
        try {
            Move-Item -LiteralPath $fullPath -Destination $Quarantine -Force
        }
        catch {
            Write-Host "  FAILED to move $relPath : $($_.Exception.Message)" -ForegroundColor Red
        }
    }
    else {
        Write-Host "Not found (already moved/clean?): $relPath" -ForegroundColor DarkGray
    }
}

Write-Host ""
Write-Host "Cleanup pass complete." -ForegroundColor Green
