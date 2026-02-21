# watch_heartbeat.ps1
# Simple 1-second heartbeat monitor for ES Paper Trader

$ErrorActionPreference = "SilentlyContinue"

# Folder where this script lives
$root   = Split-Path -Parent $MyInvocation.MyCommand.Path
$hbPath = Join-Path $root "run\heartbeat.txt"

Write-Host "=== ES Paper Trader – Heartbeat Monitor ==="
Write-Host "Watching: $hbPath"
Write-Host "Ctrl+C to exit."
Write-Host ""

# Wait for the file to exist
if (!(Test-Path $hbPath)) {
    Write-Host "Heartbeat file not found yet. Waiting for bot to create it..."
    while (!(Test-Path $hbPath)) {
        Start-Sleep -Seconds 1
    }
    Write-Host "Heartbeat file detected. Streaming updates..."
    Write-Host ""
}

# Main loop: read the JSON file once per second and print a summary
while ($true) {
    try {
        $raw = Get-Content $hbPath -Raw
        if ($raw.Trim()) {
            $hb = $raw | ConvertFrom-Json

            $ts    = $hb.ts
            $state = $hb.state
            $idle  = $hb.idle_reason
            $net   = $hb.net_qty
            $dayR  = "{0:N2}" -f $hb.day_R
            $weekR = "{0:N2}" -f $hb.week_R

            # placeholders for now; we can wire px/bars from Python later
            $px   = 0
            $bars = 0

            Write-Host ("{0} | {1} | {2} | pos={3} | R={4} | weekR={5} | px={6} | bars={7}" -f `
                $ts, $state, $idle, $net, $dayR, $weekR, $px, $bars)
        }
    } catch {
        Write-Host "Error reading heartbeat: $_"
    }

    Start-Sleep -Seconds 1
}
