# Stop ES bot + watchdog
Get-CimInstance Win32_Process -Filter 'Name="python.exe"' |
  Where-Object { $_.CommandLine -match 'watchdog_single\.py|paper_trader\.py' } |
  ForEach-Object { try { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue } catch {} }

Remove-Item "$PSScriptRoot\paper_trader.lock.client111" -Force -ErrorAction SilentlyContinue
Write-Host "ES Paper Bot stopped."