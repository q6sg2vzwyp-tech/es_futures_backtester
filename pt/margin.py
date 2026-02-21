# Auto-extracted from paper_trader.py (Margin / Funds Snapshot)
# Source of truth: pt/margin.py

# ========================= Margin / Funds Snapshot =========================
from dataclasses import dataclass, field
@dataclass
class MarginSnap:
    per_contract_init: float = 0.0   # dollars required to open 1 contract (init margin)
    available_funds:   float = 0.0   # dollar headroom for trading
    last_err:          str   = ""
    last_funds_ts:     float = 0.0
    last_margin_ts:    float = 0.0

margin_snap = MarginSnap()

def refresh_available_funds(ib=None):
    import time
    now = time.time()
    # throttle to once every 20s
    if now - margin_snap.last_funds_ts < 20:
        return
    try:
        # Prefer AccountSummary -> AvailableFunds
        acct = None
        if hasattr(ib, "wrapper") and getattr(ib.wrapper, "accounts", None):
            acct = ib.wrapper.accounts[0]
        vals = []
        try:
            vals = ib.accountSummary() if ib else []
        except Exception:
            vals = []
        got = False
        for v in vals or []:
            if acct and getattr(v, "account", None) != acct:
                continue
            if getattr(v, "tag", "") in ("AvailableFunds",):
                margin_snap.available_funds = float(v.value)
                got = True
                break
        if not got:
            # fallback: ib.accountValues()
            try:
                for v in (ib.accountValues() if ib else []):
                    if getattr(v, "tag", "") in ("AvailableFunds","ExcessLiquidity"):
                        margin_snap.available_funds = float(v.value)
                        got = True
                        break
            except Exception:
                pass
        margin_snap.last_funds_ts = now
    except Exception as e:
        margin_snap.last_err = f"refresh_available_funds: {e}"

def refresh_per_contract_margin(ib=None, con=None):
    import time
    now = time.time()
    # throttle to once every 8 minutes
    if now - margin_snap.last_margin_ts < 480:
        return
    try:
        from ib_insync import MarketOrder
        if ib and con:
            o = MarketOrder("BUY", 1)
            o.whatIf = True
            st = ib.whatIfOrder(con, o)
            if st and getattr(st, "initMarginChange", None):
                margin_snap.per_contract_init = abs(float(st.initMarginChange))
            elif st and getattr(st, "maintMarginChange", None):
                margin_snap.per_contract_init = abs(float(st.maintMarginChange))
            else:
                if margin_snap.per_contract_init <= 0.0:
                    margin_snap.per_contract_init = 15000.0
        else:
            if margin_snap.per_contract_init <= 0.0:
                margin_snap.per_contract_init = 15000.0
        margin_snap.last_margin_ts = now
    except Exception as e:
        margin_snap.last_err = f"refresh_per_contract_margin: {e}"

"""
ES Paper Trader (IBKR + ib_insync)
- Paper-only safety by default
- Thompson Sampling learner (shadow/advisory/control)
- Risk profiles: balanced/aggressive/conservative
- OCO protection builder & audit (orphan sweeps, sibling cancel)
- VWAP session + optional short guards (VWAP buffer + lower-high)
- Session cutovers (multi) + persistent day guard
- Version-safe IB PnL subscription + NetLiq via accountValueEvent
- 24/5 trading window by default (no TOD blackouts unless provided)
- News kill switch: file flag + optional TOD windows + IBKR news bulletins
- 1-second JSON heartbeats with explicit idle reasons & RT status/age/queue
- Robust RT→Polling fallback (5s bars via historical polling)

NEW (this build):
- Error logging, market-data warmup, TRADES-starved→MIDPOINT resubscribe
- Parameter meta-learning (Thompson) over per-trade parameter sets
- Persistent save/load of learners to JSON (auto-save after each flat)
- Auto self-backup of this script into .\\backups\\ on startup
- ClientId-aware order management (ignores ChartTrader/manual orders)
- Parent LIMIT → MARKET promotion after configurable time
- Slower base cadence for IBKR + adaptive cadence scaling on stress events
"""

import sys, os, time, json, math, random, argparse, datetime as dt, re, traceback, threading, shutil
from typing import Optional, List, Dict, Any, Tuple

# 3rd party
from ib_insync import IB, Future, Contract, LimitOrder, StopOrder, MarketOrder, Trade

# ---- Global: active API clientId (set after connect) ----
ACTIVE_CLIENT_ID: Optional[int] = None

# ---------- Utilities ----------
from pt.utils_time import utc_now_str, ct_now, parse_hhmm
from pt.utils_math import clamp, ticks_to_price_delta, round_to_tick

# ---------- Logging ----------
from pt.logging import log as log
