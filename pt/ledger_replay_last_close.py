import csv, json, os, datetime as dt
import trade_bridge

ev_path = os.path.join("results", "trade_events.csv")

last = None
with open(ev_path, "r", encoding="utf-8") as f:
    for line in f:
        if ",close_realized_pnl," in line:
            last = line.strip()

if not last:
    raise SystemExit("No close_realized_pnl found in trade_events.csv")

parts = list(csv.reader([last]))[0]
ts_s = parts[0]
side_raw = parts[5]      # BUY/SELL in event rows
qty_s = parts[6]
extra = json.loads(parts[-1])

trade_ts = dt.datetime.fromisoformat(ts_s)

# normalize side to LONG/SHORT
side_norm = "LONG" if str(side_raw).strip().upper() in ("BUY","LONG") else "SHORT"

row = {
    "timestamp": trade_ts.isoformat(timespec="seconds"),
    "side": side_norm,
    "qty": str(int(float(qty_s))) if qty_s else "1",
    "entry_px": f"{float(extra.get('entry_px', 0.0)):.2f}",
    "exit_px": f"{float(extra.get('exit_px', 0.0)):.2f}",
    "pnl": f"{float(extra.get('pnl_delta', 0.0)):.2f}",
    "R": "",  # only trade_bridge close path computes this; replay is for plumbing validation
    "tags": extra.get("tags","") + ";ledger_replay=1",
}

trade_bridge._canon_trades_writer_append(trade_bridge.TRADES_LEDGER_PATH, row)
print("OK: replay appended ->", trade_bridge.TRADES_LEDGER_PATH)
print("ts =", row["timestamp"])
