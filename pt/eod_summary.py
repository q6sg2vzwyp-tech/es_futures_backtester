#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eod_summary.py

Quick end-of-day + end-of-week summary for ES Paper Trader:

- Reads results/trades.csv
    * Figures out the most recent trade date.
    * PER-DAY (EOD) for that date:
        - trades, wins, losses
        - win rate
        - total PnL (USD)
        - avg R / trade
        - shows last 5 trades for that day

    * PER-WEEK (EOW) for the week ending on that latest date:
        - date range [start .. end]
        - trades, wins, losses
        - win rate
        - total PnL (USD)
        - avg R / trade
        - per-day PnL & total R table

- Reads run/heartbeat.txt
    * Assumes each heartbeat is either:
        - A JSON line (current style), OR
        - A text block separated by blank lines (old style)
    * Grabs the last entry (final state)
    * Prints:
        - timestamp
        - state, idle_reason, caps
        - net_qty, last_px
        - day_R, week_R, running_pnl_today
        - trades_today, total_trades
        - shadow_* fields if present

    * Also scans the entire heartbeat file for "today" (latest trade
      day) to list all caps seen that day.
"""

import csv
import json
import datetime as dt
from pathlib import Path
from statistics import mean
from typing import List, Dict, Tuple, Optional, Any


BASE_DIR = Path(__file__).resolve().parent
TRADES_PATH = BASE_DIR / "results" / "trades.csv"
HB_PATH = BASE_DIR / "run" / "heartbeat.txt"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_iso_datetime(s: str) -> Optional[dt.datetime]:
    if not s:
        return None
    s = s.strip()
    try:
        # Handles '2025-11-20T13:32:51-06:00' style
        return dt.datetime.fromisoformat(s)
    except Exception:
        return None


def safe_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Trades loading
# ---------------------------------------------------------------------------

def load_trades_with_dates(path: Path) -> Tuple[Optional[str], List[Tuple[dt.date, Dict[str, Any]]]]:
    """
    Load all trades from trades.csv and attach a date parsed from a
    timestamp-like column.

    Returns:
        (ts_col_name or None, list of (date, row))

    If no rows or no timestamp column can be parsed, returns (None, []).
    """
    if not path.exists():
        print(f"[WARN] trades.csv not found at {path}")
        return None, []

    rows: List[Dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        print("[INFO] trades.csv is empty (no trades).")
        return None, []

    # Try to detect trade date from close_time (or similar)
    ts_col_candidates = ["close_time", "exit_time", "close_ts", "timestamp", "time"]
    ts_col = None
    for c in ts_col_candidates:
        if c in rows[0]:
            ts_col = c
            break

    if ts_col is None:
        print("[WARN] No timestamp column (close_time/exit_time/...) found in trades.csv.")
        return None, []

    dated_rows: List[Tuple[dt.date, Dict[str, Any]]] = []
    for r in rows:
        ts = parse_iso_datetime(r.get(ts_col, "") or "")
        if ts is None:
            continue
        dated_rows.append((ts.date(), r))

    if not dated_rows:
        print("[WARN] Could not parse any trade timestamps.")
        return None, []

    return ts_col, dated_rows


def load_trades_for_latest_day(path: Path) -> Tuple[Optional[dt.date], List[Dict[str, Any]], Optional[str]]:
    """
    Returns:
        (latest_date, list_of_rows_for_that_date, ts_col_name)
    """
    ts_col, dated_rows = load_trades_with_dates(path)
    if ts_col is None or not dated_rows:
        return None, [], ts_col

    latest_date = max(d for d, _ in dated_rows)
    day_rows = [r for d, r in dated_rows if d == latest_date]
    return latest_date, day_rows, ts_col


def load_trades_for_week_ending(
    path: Path,
    ts_col: Optional[str],
    latest_date: Optional[dt.date],
) -> Tuple[Optional[dt.date], Optional[dt.date], List[Dict[str, Any]]]:
    """
    Build a "week" as the 7-calendar-day window ending on latest_date.

    Returns:
        (week_start_date, week_end_date, list_of_rows_in_that_range)
    """
    if ts_col is None or latest_date is None:
        return None, None, []

    _, dated_rows = load_trades_with_dates(path)
    if not dated_rows:
        return None, None, []

    week_end = latest_date
    week_start = week_end - dt.timedelta(days=6)

    week_rows: List[Dict[str, Any]] = [
        r for d, r in dated_rows if week_start <= d <= week_end
    ]

    if not week_rows:
        return None, None, []

    return week_start, week_end, week_rows


# ---------------------------------------------------------------------------
# Trades summaries (day + week)
# ---------------------------------------------------------------------------

def summarize_trades_day(day: Optional[dt.date], rows: List[Dict[str, Any]], ts_col: Optional[str]):
    if day is None or not rows:
        print("\n[TRADES] No trades to summarize.")
        return

    pnls = [safe_float(r.get("pnl") or r.get("PnL") or r.get("pnl_usd"), 0.0) for r in rows]
    Rs = [safe_float(r.get("R") or r.get("r"), 0.0) for r in rows]

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    n_trades = len(pnls)
    n_wins = len(wins)
    n_losses = len(losses)
    total_pnl = sum(pnls)
    win_rate = (n_wins / n_trades * 100.0) if n_trades > 0 else 0.0
    avg_R = mean(Rs) if Rs else 0.0

    print("\n================== TRADES (Most Recent Day) ==================")
    print(f"Date          : {day.isoformat()}")
    print(f"Trades        : {n_trades} (wins={n_wins}, losses={n_losses})")
    print(f"Win rate      : {win_rate:.2f}%")
    print(f"Total PnL     : {total_pnl:.2f} USD")
    print(f"Avg R/trade   : {avg_R:.3f}")
    print("==============================================================")

    # Show last 5 trades for that day
    print("\nLast 5 trades (most recent day):")

    if ts_col is not None:
        rows_sorted = sorted(
            rows,
            key=lambda r: parse_iso_datetime(r.get(ts_col, "") or "") or dt.datetime.min,
        )
    else:
        rows_sorted = rows

    for r in rows_sorted[-5:]:
        ts = r.get(ts_col, "") if ts_col else ""
        side = r.get("side", r.get("direction", ""))
        qty = r.get("qty", r.get("size", ""))
        R = safe_float(r.get("R") or r.get("r"), 0.0)
        pnl = safe_float(r.get("pnl") or r.get("PnL") or r.get("pnl_usd"), 0.0)
        print(f"- {ts} | {side} {qty} | R={R:.3f} | PnL={pnl:.2f} USD")


def summarize_trades_week(
    week_start: Optional[dt.date],
    week_end: Optional[dt.date],
    rows: List[Dict[str, Any]],
    ts_col: Optional[str],
):
    """
    Weekly summary for the 7-day window [week_start .. week_end].
    """
    if week_start is None or week_end is None or not rows:
        print("\n[WEEK] No weekly trades to summarize.")
        return

    pnls = [safe_float(r.get("pnl") or r.get("PnL") or r.get("pnl_usd"), 0.0) for r in rows]
    Rs = [safe_float(r.get("R") or r.get("r"), 0.0) for r in rows]

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    n_trades = len(pnls)
    n_wins = len(wins)
    n_losses = len(losses)
    total_pnl = sum(pnls)
    win_rate = (n_wins / n_trades * 100.0) if n_trades > 0 else 0.0
    avg_R = mean(Rs) if Rs else 0.0

    print("\n================== WEEKLY SUMMARY (EOW) ======================")
    print(f"Week range    : {week_start.isoformat()}  ->  {week_end.isoformat()}")
    print(f"Trades        : {n_trades} (wins={n_wins}, losses={n_losses})")
    print(f"Win rate      : {win_rate:.2f}%")
    print(f"Total PnL     : {total_pnl:.2f} USD")
    print(f"Avg R/trade   : {avg_R:.3f}")
    print("==============================================================")

    # Per-day breakdown within that week
    # Re-load with dates so we can group.
    ts_col2, dated_rows = load_trades_with_dates(TRADES_PATH)
    if ts_col2 is None:
        return

    per_day_pnl: Dict[dt.date, float] = {}
    per_day_R: Dict[dt.date, float] = {}
    per_day_n: Dict[dt.date, int] = {}

    for d, r in dated_rows:
        if week_start <= d <= week_end:
            pnl = safe_float(r.get("pnl") or r.get("PnL") or r.get("pnl_usd"), 0.0)
            R = safe_float(r.get("R") or r.get("r"), 0.0)
            per_day_pnl[d] = per_day_pnl.get(d, 0.0) + pnl
            per_day_R[d] = per_day_R.get(d, 0.0) + R
            per_day_n[d] = per_day_n.get(d, 0) + 1

    if not per_day_pnl:
        return

    print("\nPer-day PnL / R within this week:")
    print("--------------------------------------------------------------")
    print(f"{'Date':<12} {'Trades':>6} {'Total R':>10} {'Total PnL (USD)':>18}")
    print("--------------------------------------------------------------")
    for d in sorted(per_day_pnl.keys()):
        print(
            f"{d.isoformat():<12} "
            f"{per_day_n.get(d, 0):>6d} "
            f"{per_day_R.get(d, 0.0):>+10.3f} "
            f"{per_day_pnl.get(d, 0.0):>18.2f}"
        )
    print("--------------------------------------------------------------")


# ---------------------------------------------------------------------------
# Heartbeat summary
# ---------------------------------------------------------------------------

def parse_last_heartbeat_entry(path: Path):
    """
    Supports two formats:

    1) JSON-per-line (current style):
       {"bars": 2013, "bayes_source": "...", ...}

    2) Old text blocks separated by blank lines.

    Returns either:
      - dict (if JSON)
      - raw string block (if text)
    """
    if not path.exists():
        print(f"[WARN] heartbeat.txt not found at {path}")
        return None

    text = path.read_text(encoding="utf-8", errors="ignore")

    # Try JSON-per-line first: last non-empty line that parses as JSON
    lines = [ln for ln in text.splitlines() if ln.strip()]
    for ln in reversed(lines):
        s = ln.strip()
        if not s:
            continue
        if s.startswith("{") and s.endswith("}"):
            try:
                data = json.loads(s)
                return data
            except Exception:
                # Fall back to block style
                break

    # Fallback: block style, last non-empty block
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    if not blocks:
        print("[WARN] heartbeat.txt is empty.")
        return None

    return blocks[-1]


def scan_caps_for_today(hb_path: Path, trade_day: Optional[dt.date]):
    """
    Scan the entire heartbeat file and collect all distinct caps
    seen on the given trade_day (date).
    """
    if trade_day is None or not hb_path.exists():
        return set()

    caps_seen = set()

    with hb_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                hb = json.loads(line)
            except Exception:
                continue

            ts_str = hb.get("timestamp") or hb.get("ts")
            if not ts_str:
                continue

            ts = parse_iso_datetime(ts_str)
            if ts is None or ts.date() != trade_day:
                continue

            caps = hb.get("caps") or []
            for c in caps:
                caps_seen.add(str(c))

    return caps_seen


def summarize_heartbeat(entry):
    if entry is None:
        print("\n[HEARTBEAT] No heartbeat entry to summarize.")
        return

    # JSON dict style (current format)
    if isinstance(entry, dict):
        hb = entry
        print("\n================= HEARTBEAT (Final State) ====================")

        def get(key, default="-"):
            return hb.get(key, default)

        print(f"timestamp      : {get('timestamp')}")
        print(f"state          : {get('state')}")
        print(f"idle_reason    : {get('idle_reason')}")
        print(f"caps           : {get('caps')}")
        print(f"net_qty        : {get('net_qty')}")
        print(f"pos_state      : {get('pos_state')}")
        print(f"last_px        : {get('last_px')}")
        print(f"running_pnl    : {get('running_pnl_today')}")
        print(f"day_R          : {get('day_R')}")
        print(f"week_R         : {get('week_R')}")
        print(f"trades_today   : {get('trades_today')}")
        print(f"total_trades   : {get('total_trades')}")

        # Shadow stats (if wired in later)
        shadow_keys = [
            "shadow_long_R",
            "shadow_long_pnl",
            "shadow_long_entry_px",
            "shadow_short_R",
            "shadow_short_pnl",
            "shadow_R",
            "shadow_pnl",
        ]
        printed_shadow = False
        for k in shadow_keys:
            if k in hb:
                if not printed_shadow:
                    print("\nShadow (if wired):")
                    printed_shadow = True
                print(f"  {k:20}: {hb[k]}")

        print("\n----------------- Raw last heartbeat JSON --------------------")
        print(json.dumps(hb, indent=2, sort_keys=True))
        print("==============================================================")

    # Old text block style
    else:
        block = str(entry)
        print("\n================= HEARTBEAT (Final State) ====================")
        print(block)
        print("==============================================================")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("ES Paper Trader - End of Day / End of Week Summary")
    print("==================================================")

    # 1) EOD: Trades for the most recent day
    day, day_trades, ts_col = load_trades_for_latest_day(TRADES_PATH)
    summarize_trades_day(day, day_trades, ts_col)

    # 1b) EOW: Weekly summary ending on that day
    week_start, week_end, week_trades = load_trades_for_week_ending(
        TRADES_PATH, ts_col, day
    )
    summarize_trades_week(week_start, week_end, week_trades, ts_col)

    # 1c) Caps that were hit today (from heartbeat history)
    caps_today = scan_caps_for_today(HB_PATH, day)
    if caps_today:
        print("\nCaps seen today (from heartbeat):", ", ".join(sorted(caps_today)))
    else:
        print("\nCaps seen today (from heartbeat): none")

    # 2) Heartbeat (final state)
    hb_entry = parse_last_heartbeat_entry(HB_PATH)
    summarize_heartbeat(hb_entry)


if __name__ == "__main__":
    main()

