#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
migrate_trades.py

One-time cleanup for results/trades.csv:

- Backs up the existing file to trades_legacy_backup.csv
- Writes a new trades.csv containing ONLY "good" trades:
    * side in {LONG, SHORT}
    * pnl_usd is numeric
    * R is numeric
"""

import os
import csv
import math
from typing import Dict, List

BASE_DIR = os.path.abspath(os.path.dirname(__file__) or ".")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
SRC_PATH = os.path.join(RESULTS_DIR, "trades.csv")
BACKUP_PATH = os.path.join(RESULTS_DIR, "trades_legacy_backup.csv")


def safe_float(s: str):
    try:
        return float(s)
    except Exception:
        return float("nan")


def main() -> None:
    if not os.path.exists(SRC_PATH):
        print(f"[ERR] trades.csv not found: {SRC_PATH}")
        return

    # Load old file
    with open(SRC_PATH, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows: List[Dict[str, str]] = list(reader)

    print(f"[INFO] Loaded {len(rows)} rows from trades.csv")

    # Backup original
    if not os.path.exists(BACKUP_PATH):
        with open(BACKUP_PATH, "w", encoding="utf-8", newline="") as f:
            if rows:
                writer = csv.DictWriter(f, fieldnames=reader.fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        print(f"[INFO] Backed up original to {BACKUP_PATH}")
    else:
        print(f"[WARN] Backup {BACKUP_PATH} already exists; not overwriting.")

    # Filter "good" trades
    good: List[Dict[str, str]] = []
    for r in rows:
        side = (r.get("side", "") or "").strip().upper()
        pnl = safe_float(r.get("pnl_usd", ""))
        R = safe_float(r.get("R", ""))
        if side in ("LONG", "SHORT") and not math.isnan(pnl) and not math.isnan(R):
            good.append(r)

    print(f"[INFO] Keeping {len(good)} 'good' trades; discarding {len(rows) - len(good)} legacy rows")

    # If there are no good trades yet, bail out
    if not good:
        print("[WARN] No good trades detected; not rewriting trades.csv")
        return

    # Rewrite trades.csv with only good rows, preserving original header order
    fieldnames = ["ts", "side", "qty", "entry_px", "exit_px", "pnl_usd", "R"]
    with open(SRC_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in good:
            writer.writerow({
                "ts": r.get("ts", ""),
                "side": r.get("side", ""),
                "qty": r.get("qty", ""),
                "entry_px": r.get("entry_px", ""),
                "exit_px": r.get("exit_px", ""),
                "pnl_usd": r.get("pnl_usd", ""),
                "R": r.get("R", ""),
            })

    print("[INFO] Rewrote trades.csv with cleaned trades only.")


if __name__ == "__main__":
    main()

