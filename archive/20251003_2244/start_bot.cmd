"C:\Users\owner\AppData\Local\Programs\Python\Python313\python.exe" -X faulthandler -u paper_trader.py ^
  --host 127.0.0.1 --port 7497 --clientId 19 ^
  --symbol ES --expiry 202509 --exchange CME ^
  --tp-ladder "1.0R:50,2.0R:50" --place-orders --debug-regime ^
  --alpha 0.6 --epsilon 0.05 ^
  --reward-mode linear --reward-penalty-time 0.10 --reward-penalty-dd 0.20 --reward-penalty-slip 0.05 --reward-regime-bonus 0.02 ^
  --persist-chooser --chooser-model-path models\linucb_state.json
