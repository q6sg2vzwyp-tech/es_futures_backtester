import trade_bridge, datetime as dt
row = {
  "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
  "side": "LONG",
  "qty": "0",
  "entry_px": "0.00",
  "exit_px": "0.00",
  "pnl": "0.00",
  "R": "0.000000",
  "tags": "ledger_selftest=1",
}
trade_bridge._canon_trades_writer_append(trade_bridge.TRADES_LEDGER_PATH, row)
print("OK: wrote ledger selftest row")
