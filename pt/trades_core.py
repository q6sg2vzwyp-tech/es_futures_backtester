#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
trades_core.py

Centralized helpers for ES Paper Trader:

- Robust trades.csv loader (handles old / partial rows)
- R computation and normalization
- Lifetime / daily stats
- Safe trade logging (always logs R)
- NEW: side-specific stats (LONG vs SHORT) helpers
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict
import csv
import os
from datetime import datetime, date
import shutil
import re

# ---- Defaults for ES ----
DEFAULT_RISK_TICKS = 30         # your per-trade risk in ticks
DEFAULT_TICK_VALUE = 12.5       # ES tick value in USD


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    timestamp: str
    side: str
    qty: int
    pnl: float
    R: float
    entry_px: Optional[float] = None
    exit_px: Optional[float] = None
    tags: str = ""
    dt: Optional[datetime] = None


@dataclass
class LifetimeStats:
    trades: int
    wins: int
    losses: int
    flats: int
    win_rate: float
    avg_R: float
    realized_pnl: float


@dataclass
class DailyStats:
    trades_today: int
    day_R: float
    pnl_today: float


TRADES_HEADER = [
    "timestamp",
    "side",
    "qty",
    "entry_px",
    "exit_px",
    "pnl",
    "R",
    "tags",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_float(s, default=0.0):
    """
    Robust float parser:
    - Handles None / empty
    - Handles '123.45 USD' (takes first token, strips non-numeric chars)
    - Handles commas, signs, exponent, etc.
    - Falls back to default on failure
    """
    try:
        if s is None:
            return default
        if isinstance(s, (int, float)):
            return float(s)

        s = str(s).strip()
        if not s:
            return default

        # Take first token before any whitespace (e.g., "123.45 USD" -> "123.45")
        s = s.split()[0]

        # Keep only characters that make sense in a float
        cleaned = re.sub(r"[^0-9eE\+\-\.]", "", s)
        if not cleaned:
            return default

        return float(cleaned)
    except Exception:
        return default


def _parse_int(s, default=0):
    try:
        if s is None:
            return default
        if isinstance(s, int):
            return s
        return int(str(s).strip())
    except Exception:
        return default


def _compute_R(
    pnl: float,
    qty: int,
    risk_ticks: float = DEFAULT_RISK_TICKS,
    tick_value: float = DEFAULT_TICK_VALUE,
) -> float:
    denom = risk_ticks * tick_value * max(qty, 1)
    if denom == 0:
        return 0.0
    return pnl / denom


def _parse_timestamp(ts: str) -> Optional[datetime]:
    ts = (ts or "").strip()
    if not ts or ts == "?":
        return None

    # try ISO-like formats commonly used
    fmts = [
        "%Y-%m-%dT%H:%M:%S%z",   # 2025-12-01T14:44:04-06:00
        "%Y-%m-%dT%H:%M:%S",     # 2025-12-01T10:49:54
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(ts, fmt)
        except Exception:
            continue

    # last resort
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public: safe loader & writer
# ---------------------------------------------------------------------------

def load_trades_safe(
    csv_path: str,
    risk_ticks: float = DEFAULT_RISK_TICKS,
    tick_value: float = DEFAULT_TICK_VALUE,
) -> List[TradeRecord]:
    """
    Load trades.csv robustly.

    - Handles missing columns
    - Computes R if missing
    - Normalizes side, qty, pnl
    - Sorts by timestamp (unknown timestamps last)
    """
    trades: List[TradeRecord] = []

    if not os.path.exists(csv_path):
        return trades

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        try:
            reader = csv.DictReader(f)
        except Exception:
            # file is broken / non-CSV
            return trades

        for row in reader:
            if not row:
                continue

            ts = (row.get("timestamp") or "").strip() or "?"
            side = (row.get("side") or "").strip().upper() or "?"
            qty = _parse_int(row.get("qty", "1"), default=1)
            pnl = _parse_float(row.get("pnl", "0.0"), default=0.0)
            entry_px = row.get("entry_px")
            exit_px = row.get("exit_px")
            R_raw = (row.get("R") or "").strip()
            tags = (row.get("tags") or "").strip()

            entry_px_f: Optional[float] = None
            exit_px_f: Optional[float] = None
            if entry_px not in (None, "", "-"):
                entry_px_f = _parse_float(entry_px, default=None)
            if exit_px not in (None, "", "-"):
                exit_px_f = _parse_float(exit_px, default=None)

            # Compute R if missing or garbage
            if R_raw in ("", "-", "nan", "NaN", None):
                R_val = _compute_R(pnl, qty, risk_ticks, tick_value)
            else:
                R_val = _parse_float(
                    R_raw,
                    default=_compute_R(pnl, qty, risk_ticks, tick_value),
                )

            dt_obj = _parse_timestamp(ts)

            trades.append(
                TradeRecord(
                    timestamp=ts,
                    side=side,
                    qty=qty,
                    pnl=pnl,
                    R=R_val,
                    entry_px=entry_px_f,
                    exit_px=exit_px_f,
                    tags=tags,
                    dt=dt_obj,
                )
            )

    # Sort: valid timestamps first (ascending by actual time), unknown last
    def _sort_key(tr: TradeRecord):
        # Unknown timestamp -> put at the end
        if tr.dt is None:
            return (1, 0.0)
        try:
            # Works for both naive and tz-aware datetimes
            ts = tr.dt.timestamp()
        except Exception:
            return (1, 0.0)
        # Known timestamp -> group 0, then by epoch seconds
        return (0, ts)

    trades.sort(key=_sort_key)
    return trades


def ensure_trades_header(csv_path: str) -> None:
    """
    If trades.csv doesn't exist or has no header, create it with TRADES_HEADER.
    """
    dir_name = os.path.dirname(csv_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(TRADES_HEADER)
        return

    # Check first line for header
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        first_line = f.readline()
        if not first_line:
            # empty file
            with open(csv_path, "w", encoding="utf-8", newline="") as fw:
                writer = csv.writer(fw)
                writer.writerow(TRADES_HEADER)
            return

        # simple heuristic: if it doesn't contain "timestamp", rewrite
        if "timestamp" not in first_line:
            # backup original
            backup = csv_path + ".no_header.bak"
            shutil.copy2(csv_path, backup)
            rest = f.read()
            with open(csv_path, "w", encoding="utf-8", newline="") as fw:
                writer = csv.writer(fw)
                writer.writerow(TRADES_HEADER)
                if rest:
                    fw.write(rest)


def write_trades(
    csv_path: str,
    trades: List[TradeRecord],
    backup: bool = True,
) -> None:
    """
    Overwrite trades.csv with normalized rows.
    """
    if backup and os.path.exists(csv_path):
        shutil.copy2(csv_path, csv_path + ".bak")

    dir_name = os.path.dirname(csv_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRADES_HEADER)
        writer.writeheader()
        for t in trades:
            writer.writerow({
                "timestamp": t.timestamp,
                "side": t.side,
                "qty": t.qty,
                "entry_px": "" if t.entry_px is None else t.entry_px,
                "exit_px": "" if t.exit_px is None else t.exit_px,
                "pnl": t.pnl,
                "R": t.R,
                "tags": t.tags,
            })


# ---------------------------------------------------------------------------
# Public: stats
# ---------------------------------------------------------------------------

def compute_lifetime_stats(trades: List[TradeRecord]) -> LifetimeStats:
    n = len(trades)
    if n == 0:
        return LifetimeStats(
            trades=0,
            wins=0,
            losses=0,
            flats=0,
            win_rate=0.0,
            avg_R=0.0,
            realized_pnl=0.0,
        )

    eps = 1e-9
    wins = sum(1 for t in trades if t.pnl > eps)
    losses = sum(1 for t in trades if t.pnl < -eps)
    flats = n - wins - losses

    win_rate = (wins / max(n, 1)) * 100.0
    avg_R = sum(t.R for t in trades) / max(n, 1)
    realized_pnl = sum(t.pnl for t in trades)

    return LifetimeStats(
        trades=n,
        wins=wins,
        losses=losses,
        flats=flats,
        win_rate=win_rate,
        avg_R=avg_R,
        realized_pnl=realized_pnl,
    )


def compute_daily_stats(
    trades: List[TradeRecord],
    today: Optional[date] = None,
) -> DailyStats:
    if today is None:
        today = date.today()

    todays: List[TradeRecord] = []
    for t in trades:
        if t.dt is None:
            continue
        if t.dt.date() == today:
            todays.append(t)

    trades_today = len(todays)
    day_R = sum(t.R for t in todays)
    pnl_today = sum(t.pnl for t in todays)

    return DailyStats(
        trades_today=trades_today,
        day_R=day_R,
        pnl_today=pnl_today,
    )


# ---------------------------------------------------------------------------
# NEW: side-specific stats (LONG vs SHORT, etc.)
# ---------------------------------------------------------------------------

def compute_side_lifetime_stats(
    trades: List[TradeRecord],
) -> Dict[str, LifetimeStats]:
    """
    Compute lifetime stats per side.

    Returns a dict keyed by side (e.g. "LONG", "SHORT", "OTHER"),
    where each value is a LifetimeStats for that subset of trades.
    """
    buckets: Dict[str, List[TradeRecord]] = {}

    for t in trades:
        side = (t.side or "?").upper()
        if side not in ("LONG", "SHORT"):
            side_key = "OTHER"
        else:
            side_key = side
        buckets.setdefault(side_key, []).append(t)

    side_stats: Dict[str, LifetimeStats] = {}
    for side_key, sub_trades in buckets.items():
        side_stats[side_key] = compute_lifetime_stats(sub_trades)

    return side_stats


def compute_side_daily_stats(
    trades: List[TradeRecord],
    today: Optional[date] = None,
) -> Dict[str, DailyStats]:
    """
    Compute daily stats per side for the given date (default = today).

    Returns a dict keyed by side (e.g. "LONG", "SHORT", "OTHER").
    """
    if today is None:
        today = date.today()

    buckets: Dict[str, List[TradeRecord]] = {}

    for t in trades:
        if t.dt is None or t.dt.date() != today:
            continue
        side = (t.side or "?").upper()
        if side not in ("LONG", "SHORT"):
            side_key = "OTHER"
        else:
            side_key = side
        buckets.setdefault(side_key, []).append(t)

    side_stats: Dict[str, DailyStats] = {}
    for side_key, sub_trades in buckets.items():
        side_stats[side_key] = compute_daily_stats(sub_trades, today=today)

    return side_stats


# ---------------------------------------------------------------------------
# Public: trade logging (to be called from paper_trader)
# ---------------------------------------------------------------------------

def log_trade(
    csv_path: str,
    timestamp: datetime,
    side: str,
    qty: int,
    entry_px: float,
    exit_px: float,
    pnl: float,
    tags: str = "",
    risk_ticks: float = DEFAULT_RISK_TICKS,
    tick_value: float = DEFAULT_TICK_VALUE,
) -> None:
    """
    Append a trade to trades.csv, computing R.

    Call this from your execution / exit logic in paper_trader.
    """
    ensure_trades_header(csv_path)

    side = side.upper().strip()
    if side not in ("LONG", "SHORT"):
        # keep something (don't explode), but mark as unknown
        side = "?"

    R_val = _compute_R(pnl, qty, risk_ticks, tick_value)

    ts_str = timestamp.isoformat()
    row = {
        "timestamp": ts_str,
        "side": side,
        "qty": qty,
        "entry_px": entry_px,
        "exit_px": exit_px,
        "pnl": pnl,
        "R": R_val,
        "tags": tags,
    }

    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRADES_HEADER)
        # header is ensured already
        writer.writerow(row)

    # Also append to immutable ledger for long-horizon learning/audit (same 8-col schema)
    ledger_path = os.path.join("results", "trades_ledger.csv")
    try:
        # Ensure header exists (do not rotate/clean; ledger is append-only)
        if (not os.path.exists(ledger_path)) or (os.path.getsize(ledger_path) == 0):
            dir_name = os.path.dirname(ledger_path)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name, exist_ok=True)
            with open(ledger_path, "w", encoding="utf-8", newline="") as f0:
                w0 = csv.writer(f0)
                w0.writerow(TRADES_HEADER)

        with open(ledger_path, "a", encoding="utf-8", newline="") as f2:
            writer2 = csv.DictWriter(f2, fieldnames=TRADES_HEADER, extrasaction="ignore")
            writer2.writerow(row)
    except Exception:
        pass

