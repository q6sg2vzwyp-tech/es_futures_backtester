#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pnl_display_core.py  (v2.5)

Pretty-print ES Paper Trader heartbeat + PnL stats.

- Uses pnl_core.compute() for all stats derived from trades.csv
- Keeps heartbeat fields (day_R, week_R, caps, meta, etc.) as-is
- Shows consistent:
    * Lifetime trades / wins / losses / flats
    * Win rate
    * Avg R/trade
    * Realized PnL
    * Today's trades / R / PnL
    * Last 3 trades (no blank side / R)
"""

from __future__ import annotations

from datetime import datetime, date
from pathlib import Path
from typing import Optional, Mapping, Any

from pnl_core import snapshot_es_pnl_and_orders as compute_pnl


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _fmt_float(val: Optional[float], digits: int = 2, fallback: str = "-") -> str:
    if val is None:
        return fallback
    try:
        return f"{float(val):.{digits}f}"
    except Exception:
        return fallback


def _resolve_today(hb: Mapping[str, Any]) -> date:
    """
    Use the heartbeat timestamp if present, otherwise local today().
    Expected formats:
      - '2025-11-28T08:43:17-06:00'
      - '2025-11-28 08:43:17'
      - '2025-11-28'
    """
    ts = (hb.get("timestamp") or "").strip()
    if not ts:
        return date.today()

    # strip timezone if present
    if "+" in ts:
        ts = ts.split("+", 1)[0]
    if "-" in ts and ts.count("-") >= 2 and "T" in ts:
        # yyyy-mm-ddT...
        d = ts.split("T", 1)[0]
    else:
        d = ts.split(" ", 1)[0]

    try:
        return datetime.strptime(d, "%Y-%m-%d").date()
    except Exception:
        return date.today()


# ---------------------------------------------------------------------------
# main render entrypoint
# ---------------------------------------------------------------------------

def render_dashboard(
    hb: Mapping[str, Any],
    hb_path: Path,
    trades_path: Path,
    version: str = "v2.5",
) -> None:
    """
    Pretty-print a full dashboard frame.

    hb        : dict-like heartbeat structure produced by hb_core
    hb_path   : Path to heartbeat.txt
    trades_path : Path to results/trades.csv
    """
    # ---- PnL stats from trades.csv ----
    today = _resolve_today(hb)
    pnl_stats: PnlStats = compute_pnl(trades_path, today)

    # ---- Header ----
    print(f"ES Paper Trader Heartbeat Dashboard ({version})")
    print("------------------------------------------")
    print(f"HB file : {hb_path}")
    print(f"Trades  : {trades_path}")
    print()

    # ---- State ----
    print("State")
    print("-----")
    print(f"Timestamp    : {hb.get('timestamp', '-')}")
    state = hb.get("state", "-")
    idle_reason = hb.get("idle_reason", "")
    print(f"State        : {state}  | idle_reason: {idle_reason}")
    print(f"Caps         : {hb.get('caps', '-')}")
    bayes_src = hb.get("bayes_source", "-")
    restart_ct = hb.get("restart_ct", "-")
    print(f"bayes_source : {bayes_src}  | restart_ct: {restart_ct}")
    print()

    # ---- Position & PnL ----
    print("Position & PnL")
    print("----------------")
    pos_state = hb.get("pos_state", "flat")
    net_qty = hb.get("net_qty", 0)
    print(f"pos_state    : {pos_state}  | net_qty: {net_qty}")
    print(f"last px      : {_fmt_float(hb.get('last_px'))}")
    entry_px = hb.get("entry_px", hb.get("avg_px"))
    print(f"entry px     : {_fmt_float(entry_px)}")
    print(f"unreal PnL   : {_fmt_float(hb.get('unreal_pnl'), 2)} USD")
    print(f"bars         : {hb.get('bars', 0)}")
    print()

    # ---- Orders ----
    print("Orders")
    print("------")
    open_orders = hb.get("open_orders", 0)
    stops = hb.get("stops", hb.get("open_stops", 0))
    limits = hb.get("limits", hb.get("open_limits", 0))
    print(f"open_orders  : {open_orders} (stops={stops}, limits={limits})")
    print(f"stop_px      : {_fmt_float(hb.get('stop_px'))}")
    print(f"target_px    : {_fmt_float(hb.get('target_px'))}")
    print()

    # ---- Performance (Today) ----
    print("Performance (Today)")
    print("-------------------")
    # Use trade-derived counts + R, but keep rails' day/week R from HB if present
    hb_day_R = hb.get("day_R")
    hb_week_R = hb.get("week_R")

    print(f"trades_today : {pnl_stats.today_trades}")
    if hb_day_R is not None:
        print(f"day_R        : {_fmt_float(hb_day_R, 3)}")
    else:
        print(f"day_R        : {_fmt_float(pnl_stats.today_R, 3)}")

    if hb_week_R is not None:
        print(f"week_R       : {_fmt_float(hb_week_R, 3)}")
    else:
        print("week_R       : -")

    print(f"PnL (today)  : {_fmt_float(pnl_stats.today_pnl, 2)} USD")
    print(f"total_trades : {pnl_stats.total}")
    print()

    # ---- Equity / Meta ----
    print("Equity / Meta")
    print("-------------")
    print(f"equity       : {_fmt_float(hb.get('equity'), 2)}")
    print(f"equity_hwm   : {_fmt_float(hb.get('equity_hwm'), 2)}")
    print(f"hwm_factor   : {_fmt_float(hb.get('hwm_factor'), 3)}")
    print(f"meta_ema_R   : {_fmt_float(hb.get('meta_ema_R'), 3)}")
    print(f"meta_aggr    : {_fmt_float(hb.get('meta_aggr'), 3)}")
    print()

    # ---- Account ----
    print("Account (from IB)")
    print("-----------------")
    print(f"acct_unreal  : {_fmt_float(hb.get('acct_unreal'), 2)}")
    print(f"acct_realized: {_fmt_float(hb.get('acct_realized'), 2)}")
    print(f"acct_netliq  : {_fmt_float(hb.get('acct_netliq'), 2)}")
    print()

    # ---- Lifetime ----
    print("Lifetime (trades.csv)")
    print("----------------------")
    print(
        f"trades       : {pnl_stats.total}  "
        f"(wins={pnl_stats.wins}, losses={pnl_stats.losses}, flats={pnl_stats.flats})"
    )
    print(f"win_rate     : {_fmt_float(pnl_stats.win_rate, 2)}%")
    print(f"avg R/trade  : {_fmt_float(pnl_stats.avg_R, 3)}")
    print(f"realized PnL : {_fmt_float(pnl_stats.realized, 2)} USD")
    print()

    # ---- Last trades ----
    print("Last trades (most recent at bottom)")
    print("-----------------------------------")
    if not pnl_stats.last:
        print(" (none)")
    else:
        for tr in pnl_stats.last:
            r_str = "-" if tr.R is None else f"{tr.R:.4f}"
            print(f" | side={tr.side:<5} pnl={tr.pnl:.2f} R={r_str}")
    print()

    # ---- Last IB Error ----
    print("Last IB Error")
    print("-------------")
    last_err = hb.get("last_ib_error") or hb.get("last_error") or "<none>"
    print(last_err)
    print()

