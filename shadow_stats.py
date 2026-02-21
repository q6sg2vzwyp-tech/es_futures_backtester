#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
shadow_stats.py

Quick-and-dirty stats for shadow round-trip trades logged by paper_trader.py.

Reads:
    results/shadow_roundtrips.csv

Each row is expected to have:
    entry_ts, exit_ts, arm, side, entry_px, exit_px,
    pnl_usd, R, open_gate, close_gate, day, week_R, meta_ema_R

Usage:
    python shadow_stats.py
    python shadow_stats.py --day 2025-12-05
"""

import os
import csv
import argparse
import datetime as dt
from collections import defaultdict

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
SHADOW_ROUNDTRIP_LOG = os.path.join(BASE_DIR, "results", "shadow_roundtrips.csv")


def parse_float(val, default=0.0):
    try:
        return float(val)
    except Exception:
        return default


def load_roundtrips(path):
    if not os.path.exists(path):
        print(f"[WARN] No shadow roundtrip file found at {path}")
        return []

    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Be defensive: require at least exit_ts & pnl
            if not row.get("exit_ts"):
                continue
            rows.append(row)

    if not rows:
        print("[WARN] No valid round-trip rows found.")
    return rows


def group_by_day(rows):
    days = defaultdict(list)
    for r in rows:
        day_str = r.get("day")
        if not day_str:
            # Fallback: derive from exit_ts
            exit_ts = r.get("exit_ts", "")
            try:
                d = dt.datetime.fromisoformat(exit_ts).date()
                day_str = d.isoformat()
            except Exception:
                day_str = "UNKNOWN"
        days[day_str].append(r)
    return days


def summarize_rows(rows):
    """
    Returns (n_trades, pnl_sum, R_sum, pnl_avg, R_avg, winrate)
    """
    n = len(rows)
    if n == 0:
        return (0, 0.0, 0.0, 0.0, 0.0, 0.0)

    pnl_sum = 0.0
    R_sum = 0.0
    wins = 0

    for r in rows:
        pnl = parse_float(r.get("pnl_usd", "0"))
        R = parse_float(r.get("R", "0"))
        pnl_sum += pnl
        R_sum += R
        if pnl > 0:
            wins += 1

    pnl_avg = pnl_sum / n
    R_avg = R_sum / n
    winrate = (wins / n) * 100.0
    return (n, pnl_sum, R_sum, pnl_avg, R_avg, winrate)


def summarize_by_arm(rows):
    per_arm = defaultdict(list)
    for r in rows:
        arm = r.get("arm") or "UNKNOWN"
        per_arm[arm].append(r)

    stats = {}
    for arm, arm_rows in per_arm.items():
        stats[arm] = summarize_rows(arm_rows)
    return stats


def format_stats(prefix, stats_tuple):
    n, pnl_sum, R_sum, pnl_avg, R_avg, winrate = stats_tuple
    return (
        f"{prefix}: trades={n}, "
        f"pnl={pnl_sum:.2f} USD, avg={pnl_avg:.2f} USD, "
        f"R_sum={R_sum:.3f}, R_avg={R_avg:.3f}, "
        f"winrate={winrate:.1f}%"
    )


def main():
    parser = argparse.ArgumentParser(description="Shadow round-trip stats")
    parser.add_argument(
        "--day",
        type=str,
        default=None,
        help="Filter to a specific day YYYY-MM-DD (matches the 'day' column)",
    )
    args = parser.parse_args()

    rows = load_roundtrips(SHADOW_ROUNDTRIP_LOG)
    if not rows:
        return

    days = group_by_day(rows)

    if args.day:
        if args.day not in days:
            print(f"[INFO] No shadow trades found for day={args.day}")
            return
        day_rows = days[args.day]
        print(f"=== Shadow Stats for {args.day} ===")
        overall = summarize_rows(day_rows)
        print(format_stats("DAY TOTAL", overall))

        per_arm = summarize_by_arm(day_rows)
        pri

