import csv, os

src = os.path.join("results", "trades.csv")
dst = os.path.join("results", "trades_ledger.csv")

# copy trades.csv into trades_ledger.csv exactly (canonical 8-col)
with open(src, "r", newline="", encoding="utf-8") as f:
    rows = [r for r in csv.reader(f) if r and any(c.strip() for c in r)]

if not rows:
    raise SystemExit("trades.csv is empty")

# ensure header
header = ["timestamp","side","qty","entry_px","exit_px","pnl","R","tags"]
if rows[0] and rows[0][0].lower().startswith("timestamp"):
    body = rows[1:]
else:
    body = rows

with open(dst, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(header)
    for r in body:
        # normalize length to 8
        rr = (r + [""]*8)[:8]
        w.writerow(rr)

print("OK: backfilled", dst, "rows=", len(body))
