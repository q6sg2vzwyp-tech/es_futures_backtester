@echo off
cd /d "C:\Users\owner\Desktop\es_futures_backtester"
py ib_paper_trader.py --place-orders --qty 1 --fast 8 --slow 20 --rsi 14 --bracket --tp-ticks 12 --sl-ticks 8 --trail-ticks 8 --entry-limit-offset 2 --entry-timeout-s 15 --pyramid-max 2 --pyramid-step-atr 0.5 --min-atr-ticks 4 --max-spread-ticks 4 --rth-only --flat-at 14:57 --verbose
pause
