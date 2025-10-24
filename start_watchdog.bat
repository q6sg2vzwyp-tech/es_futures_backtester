@echo off
setlocal

set "PY=%~dp0.venv\Scripts\python.exe"
set "SCRIPT_DIR=%~dp0"

rem === ARGS: no --risk-includes-slippage and no --commission-per-contract ===
set PAPER_TRADER_ARGS=^
 --local-symbol ESZ5 ^
 --place-orders ^
 --use-ib-pnl ^
 --learn-mode advisory ^
 --risk-profile balanced ^
 --trade-start-ct 00:00 --trade-end-ct 23:59 ^
 --startup-delay-sec 0 ^
 --gate-bbbw 0 ^
 --poll-hist-when-no-rt --poll-interval-sec 10 --rt-staleness-sec 45 --rt-starve-sec 3 ^
 --session-reset-cts 08:30,16:00,17:00 --vwap-reset-on-session ^
 --short-guard-vwap --short-guard-lower-high ^
 --pos-age-cap-sec 1200 --pos-age-minR 0.5 ^
 --hwm-stepdown --hwm-stepdown-dollars 5000 ^
 --day-guard-pct 0.025 --day-loss-cap-R 3 --max-trades-per-day 12 --max-consec-losses 3 ^
 --tick-size 0.25 --risk-ticks 12 --tp-R 1.0 --entry-slippage-ticks 2 ^
 --margin-per-contract 13200 --margin-reserve-pct 0.10

echo [RUN] PY=%PY%
echo [RUN] SCRIPT_DIR=%SCRIPT_DIR%
echo [RUN] PAPER_TRADER_ARGS=%PAPER_TRADER_ARGS%

"%PY%" -u "%SCRIPT_DIR%\watchdog_single.py" --cwd "%SCRIPT_DIR%" --target "%SCRIPT_DIR%\paper_trader.py" --tee
