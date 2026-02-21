#!/usr/bin/env python3
import csv
import os
from datetime import datetime, timedelta

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
TRADES_CSV = os.path.join(BASE_DIR, "results", "trades.csv")


def main() -> int:
    now = datetime.now()
    start = now - timedelta(days=7)

    if not os.path.exists(TRADES_CSV):
        print("No trades.csv found.")
        return 1

    rows = []
    with open(TRADES_CSV, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            ts = r.get("ts") or r.get("timestamp") or r.get("time") or ""
            if not ts:
                continue
            try:
                ts_dt = datetime.fromisoformat(ts[:19])
            except Exception:
                continue
            if ts_dt < start:
                continue
            rows.append(r)

    if not rows:
        print("No trades in last 7 days.")
        return 0

    total = len(rows)
    wins = losses = 0
    pnl_sum = 0.0
    R_sum = 0.0
    R_ct = 0

    for r in rows:
        pnl = r.get("pnl_usd") or r.get("pnl") or r.get("pnl_dollars")
        R = r.get("R") or r.get("r_mult")

        try:
            if pnl not in (None, ""):
                p = float(pnl)
                pnl_sum += p
                if p > 0:
                    wins += 1
                elif p < 0:
                    losses += 1
        except Exception:
            pass

        try:
            if R not in (None, ""):
                R_sum += float(R)
                R_ct += 1
        except Exception:
            pass

    winloss_trades = wins + losses
    winrate = (wins / winloss_trades * 100.0) if winloss_trades > 0 else None
    avg_R = (R_sum / R_ct) if R_ct > 0 else None

    print("========== Weekly Summary (last 7 days) ==========")
    print(f"Trades      : {total}")
    print(f"Wins / Loss : {wins} / {losses}")
    print(
        f"Winrate     : {winrate:.1f}% (wins / (wins+losses))"
        if winrate is not None
        else "Winrate     : -"
    )
    print(f"Total PnL   : {pnl_sum:.2f} USD")
    print(f"Avg R/trade : {avg_R:.3f}" if avg_R is not None else "Avg R/trade : -")
    print("==================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
