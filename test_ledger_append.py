import csv, json, os, datetime as dt
import trade_bridge

ev_path = os.path.join("results","trade_events.csv")

# read last close_realized_pnl event
last = None
with open(ev_path, "r", encoding="utf-8") as f:
    for line in f:
        if ",close_realized_pnl," in line:
            last = line.strip()

if not last:
    raise SystemExit("No close_realized_pnl found")

parts = list(csv.reader([last]))[0]
# columns: ts,event,trade_id,symbol,arm,side,qty,price,fill_px,order_id,reason,extra_json
ts_s = parts[0]
extra = json.loads(parts[-1])

trade_ts = dt.datetime.fromisoformat(ts_s)
side = extra.get("tags","")
# We won't re-create the whole close path; we will just append a ledger row using the same canonical helper.
row = {
    "timestamp": trade_ts.isoformat(timespec="seconds"),
    "side": "LONG",  # side label in canonical trades.csv is LONG/SHORT; this is just a test row
    "qty": "1",
    "entry_px": f"{float(extra.get('entry_px',0.0)):.2f}",
    "exit_px": f"{float(extra.get('exit_px',0.0)):.2f}",
    "pnl": f"{float(extra.get('pnl_delta',0.0)):.2f}",
    "R": "",  # close path computes this; we only test append plumbing
    "tags": extra.get("tags",""),
}

trade_bridge._canon_trades_writer_append(trade_bridge.TRADES_LEDGER_PATH, row)
print("OK: appended 1 test ledger row ->", trade_bridge.TRADES_LEDGER_PATH)
