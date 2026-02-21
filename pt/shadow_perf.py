#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
shadow_perf.py

Summarize shadow learning performance by day + arm.

Reads:
    results/shadow_trades.csv
        ts, arm, side, prev_px, last_px, shadow_pnl_usd, shadow_R, gate_reason,
        caps, day_R, week_R, meta_ema_R

Writes:
    results/shadow_perf_by_arm.csv
        day, arm, n_trades, n_win, n_loss, win_rate_pct,
        R_sum, R_avg, R_std, sharpe_like,
        pnl_usd_sum, pnl_usd_avg
"""

import csv
import os
import math
from collections import defaultdict
from statistics import mean, pstdev
from typing import Dict, List, Tuple

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
SHADOW_TRADES_CSV = os.path.join(RESULTS_DIR, "shadow_trades.csv")
SHADOW_PERF_CSV = os.path.join(RESULTS_DIR, "shadow_perf_by_arm.csv")


def safe_float(v, default=0.0):
    try:
        if v is None:
            return default
        s = str(v).strip()
        if not s:
            return default
        return float(s)
    except Exception:
        return default


def load_shadow_trades(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Shadow trades file not found: {path}")

    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def summarize_by_day_and_arm(rows: List[Dict[str, str]]) -> Dict[Tuple[str, str], Dict[str, float]]:
    """
    Group rows by (day, arm) and compute performance stats.

    day = ts[0:10]  (YYYY-MM-DD)
    """
    buckets: Dict[Tuple[str, str], Dict[str, List[float]]] = defaultdict(lambda: {
        "R": [],
        "pnl": [],
    })

    for row in rows:
        ts = row.get("ts", "")
        if len(ts) < 10:
            # Skip junk timestamps
            continue
        day = ts[:10]

        arm = row.get("arm", "").strip() or "UNKNOWN"

        shadow_R = safe_float(row.get("shadow_R"))
        shadow_pnl = safe_float(row.get("shadow_pnl_usd"))

        key = (day, arm)
        buckets[key]["R"].append(shadow_R)
        buckets[key]["pnl"].append(shadow_pnl)

    summary: Dict[Tuple[str, str], Dict[str, float]] = {}

    for key, series in buckets.items():
        day, arm = key
        Rs = series["R"]
        PnLs = series["pnl"]

        if not Rs:
            continue

        n = len(Rs)
        n_win = sum(1 for r in Rs if r > 0.0)
        n_loss = sum(1 for r in Rs if r < 0.0)
        win_rate = (n_win / n) * 100.0 if n > 0 else 0.0

        R_sum = sum(Rs)
        R_avg = mean(Rs)
        # population std dev; if zero or only 1 sample, Sharpe-ish = 0
        R_std = pstdev(Rs) if n > 1 else 0.0

        if R_std > 0:
            sharpe_like = R_avg / R_std
        else:
            sharpe_like = 0.0

        pnl_sum = sum(PnLs)
        pnl_avg = mean(PnLs) if PnLs else 0.0

        summary[key] = {
            "n_trades": n,
            "n_win": n_win,
            "n_loss": n_loss,
            "win_rate_pct": win_rate,
            "R_sum": R_sum,
            "R_avg": R_avg,
            "R_std": R_std,
            "sharpe_like": sharpe_like,
            "pnl_usd_sum": pnl_sum,
            "pnl_usd_avg": pnl_avg,
        }

    return summary


def write_summary_csv(summary: Dict[Tuple[str, str], Dict[str, float]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    fieldnames = [
        "day",
        "arm",
        "n_trades",
        "n_win",
        "n_loss",
        "win_rate_pct",
        "R_sum",
        "R_avg",
        "R_std",
        "sharpe_like",
        "pnl_usd_sum",
        "pnl_usd_avg",
    ]

    # Sort by day then arm for nicer Excel view
    sorted_keys = sorted(summary.keys(), key=lambda k: (k[0], k[1]))

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for (day, arm) in sorted_keys:
            stats = summary[(day, arm)]
            writer.writerow({
                "day": day,
                "arm": arm,
                "n_trades": stats["n_trades"],
                "n_win": stats["n_win"],
                "n_loss": stats["n_loss"],
                "win_rate_pct": f"{stats['win_rate_pct']:.2f}",
                "R_sum": f"{stats['R_sum']:.6f}",
                "R_avg": f"{stats['R_avg']:.6f}",
                "R_std": f"{stats['R_std']:.6f}",
                "sharpe_like": f"{stats['sharpe_like']:.4f}",
                "pnl_usd_sum": f"{stats['pnl_usd_sum']:.2f}",
                "pnl_usd_avg": f"{stats['pnl_usd_avg']:.2f}",
            })


def main() -> None:
    print(f"Loading shadow trades from: {SHADOW_TRADES_CSV}")
    rows = load_shadow_trades(SHADOW_TRADES_CSV)
    if not rows:
        print("No rows found in shadow_trades.csv")
        return

    print(f"Loaded {len(rows)} shadow rows; summarizing...")
    summary = summarize_by_day_and_arm(rows)
    if not summary:
        print("No valid summary buckets (check timestamps / data).")
        return

    write_summary_csv(summary, SHADOW_PERF_CSV)
    print(f"Wrote per-arm daily summary to: {SHADOW_PERF_CSV}")


if __name__ == "__main__":
    main()

