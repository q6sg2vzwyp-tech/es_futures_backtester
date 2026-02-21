#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import math
import os
from collections import Counter
from typing import List, Dict

BASE_DIR = os.path.abspath(os.path.dirname(__file__) or ".")
TRADE_LOG_CSV = os.path.join(BASE_DIR, "results", "trades.csv")


def load_trades(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        print(f"[ERR] trades.csv not found: {path}")
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def safe_float(s: str, default=float("nan")) -> float:
    try:
        return float(s)
    except Exception:
        return default


def audit_trades(rows: List[Dict[str, str]]) -> None:
    if not rows:
        print("[INFO] No trades to audit.")
        return

    print(f"[INFO] Loaded {len(rows)} rows from trades.csv")

    # Lifetime PnL from CSV
    pnl_list = [safe_float(r.get("pnl_usd", "")) for r in rows]
    pnl_sum = sum(x for x in pnl_list if not math.isnan(x))

    # Identify "new-style" trades that have a usable R and side
    clean = []
    for r in rows:
        side = (r.get("side", "") or "").strip().upper()
        R = safe_float(r.get("R", ""))
        if side in ("LONG", "SHORT") and not math.isnan(R):
            clean.append((r, R))

    clean_count = len(clean)
    wins = sum(1 for _, R in clean if R > 0)
    losses = sum(1 for _, R in clean if R < 0)
    flats = sum(1 for _, R in clean if abs(R) < 1e-9)

    avg_R = sum(R for _, R in clean) / clean_count if clean_count else float("nan")

    print("\n=== Lifetime summary (all rows) ===")
    print(f"rows_total        : {len(rows)}")
    print(f"sum(pnl_usd)      : {pnl_sum:.2f} USD")

    print("\n=== New-style trades (with side + R) ===")
    print(f"trades_new_style  : {clean_count}")
    print(f"wins              : {wins}")
    print(f"losses            : {losses}")
    print(f"flats             : {flats}")
    if not math.isnan(avg_R):
        print(f"avg_R             : {avg_R:.3f}")
    else:
        print("avg_R             : N/A")

    # Quick sanity on reasons
    reasons = Counter((r.get("reason") or "").strip() or "-" for r, _ in clean)
    print("\n=== Exit reasons (new-style trades) ===")
    for reason, cnt in reasons.most_common():
        print(f"{reason:20s} : {cnt}")

    # Show the last few trades
    print("\n=== Last 5 trades (raw) ===")
    for r in rows[-5:]:
        print(
            f"{r.get('timestamp', '')} | side={r.get('side', ''):5s} "
            f"qty={r.get('qty','')} pnl={r.get('pnl_usd','')} R={r.get('R','')} "
            f"reason={r.get('reason','')}"
        )


def main() -> None:
    rows = load_trades(TRADE_LOG_CSV)
    audit_trades(rows)


if __name__ == "__main__":
    main()

