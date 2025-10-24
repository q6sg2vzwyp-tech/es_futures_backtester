Set-Location "C:\Users\owner\Desktop\es_futures_backtester"
Remove-Item ".\paper_trader.lock.client111" -Force -ErrorAction SilentlyContinue

$py = "C:\Users\owner\AppData\Local\Programs\Python\Python313\python.exe"
$wd = ".\watchdog_single.py"

# Use the file directly
& $py $wd --cmd-file ".\profiles\single.cmdline.txt" `
  --logdir "logs" --hb-sec 5 --restart-delay-sec 10 --always-restart *>&1 |
  Tee-Object -FilePath .\logs\watchdog_stdout.txt -Append
