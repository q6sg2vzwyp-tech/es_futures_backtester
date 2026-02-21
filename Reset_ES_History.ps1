# Reset_ES_History.ps1
# Safely archive and clear ES Paper Trader history
# Base dir: C:\Users\owner\Desktop\es_futures_backtester

$base = "C:\Users\owner\Desktop\es_futures_backtester"

if (-not (Test-Path $base)) {
    Write-Host "Base directory not found: $base" -ForegroundColor Red
    exit 1
}

Set-Location $base

$timestamp   = Get-Date -Format "yyyyMMdd_HHmmss"
$archiveRoot = Join-Path $base ("archive\reset_" + $timestamp)

# Create archive folders
$archiveResults = Join-Path $archiveRoot "results"
$archiveLogs    = Join-Path $archiveRoot "logs"
$archiveLearn   = Join-Path $archiveRoot "learn"
$archiveRun     = Join-Path $archiveRoot "run"

New-Item -ItemType Directory -Force -Path $archiveResults | Out-Null
New-Item -ItemType Directory -Force -Path $archiveLogs    | Out-Null
New-Item -ItemType Directory -Force -Path $archiveLearn   | Out-Null
New-Item -ItemType Directory -Force -Path $archiveRun     | Out-Null

Write-Host "Archiving old history to: $archiveRoot" -ForegroundColor Cyan

# --- RESULTS: trades + shadow trades + bayes training set ---
$resultsDir = Join-Path $base "results"
if (Test-Path $resultsDir) {
    Get-ChildItem $resultsDir -File | ForEach-Object {
        Move-Item $_.FullName -Destination $archiveResults -Force
    }
}

# --- LOGS: es_paper, watchdog, child, learn logs, etc. ---
$logsDir = Join-Path $base "logs"
if (Test-Path $logsDir) {
    Get-ChildItem $logsDir -Recurse -File | ForEach-Object {
        Move-Item $_.FullName -Destination $archiveLogs -Force
    }
}

# --- LEARN: bandit state, bayes_best, etc. ---
$learnDir = Join-Path $base "learn"
if (Test-Path $learnDir) {
    Get-ChildItem $learnDir -File | ForEach-Object {
        Move-Item $_.FullName -Destination $archiveLearn -Force
    }
}

# --- RUN: heartbeat / any temp runtime artifacts ---
$runDir = Join-Path $base "run"
if (Test-Path $runDir) {
    Get-ChildItem $runDir -File | ForEach-Object {
        Move-Item $_.FullName -Destination $archiveRun -Force
    }
}

Write-Host "Archive complete." -ForegroundColor Green

# Re-create empty core folders the bot expects
if (-not (Test-Path $resultsDir)) { New-Item -ItemType Directory -Path $resultsDir | Out-Null }
if (-not (Test-Path $logsDir))    { New-Item -ItemType Directory -Path $logsDir    | Out-Null }
if (-not (Test-Path $learnDir))   { New-Item -ItemType Directory -Path $learnDir   | Out-Null }
if (-not (Test-Path $runDir))     { New-Item -ItemType Directory -Path $runDir     | Out-Null }

Write-Host "Reset complete. All history cleared, fresh folders ready." -ForegroundColor Yellow
Write-Host "Next step: update single.cmdline.txt and restart the watcher." -ForegroundColor Yellow
