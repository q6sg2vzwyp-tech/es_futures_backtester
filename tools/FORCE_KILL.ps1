$ErrorActionPreference = "SilentlyContinue"
Set-Location (Split-Path $PSScriptRoot -Parent)

# clear shutdown flag
Remove-Item .\run\SHUTDOWN.flag -Force -ErrorAction SilentlyContinue

# kill any python/pythonw whose command line contains paper_trader.py
Get-CimInstance Win32_Process |
  Where-Object { $_.Name -in @("python.exe","pythonw.exe") -and $_.CommandLine -match "paper_trader\.py" } |
  ForEach-Object {
    Write-Host ("[KILL] {0} PID={1} PPID={2}" -f $_.Name,$_.ProcessId,$_.ParentProcessId)
    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
  }

Start-Sleep 1

# show remaining matches
Write-Host "`n[CHECK] Remaining matches (should be none for python.exe/pythonw.exe):"
Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -match "paper_trader\.py" } |
  Select-Object Name, ProcessId, ParentProcessId, CommandLine |
  Format-Table -AutoSize
