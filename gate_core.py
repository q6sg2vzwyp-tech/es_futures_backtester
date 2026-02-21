#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gate_core.py

Entry gating / caps logic for ES Paper Trader.

- Wraps up:
    * DayRisk.gate_reason()
    * hard_caps (safety halt, Friday lockout, holiday lockout, etc.)
    * post-flat cooldown (based on last_trade_close_ts)
    * post-loss cooldown (based on day_risk.last_loss_ts)         [NEW]
    * hourly overtrade cap (based on day_risk.trades_this_hour)   [NEW, optional]

- Returns:
    * gate_reason (str or None)
    * caps (list of strings to publish in heartbeat)
    * idle_reason (string for HB)
"""

from __future__ import annotations

from typing import List, Optional
import time
import datetime as dt

from risk_core import DayRisk


def compute_gate(
    now_ct: dt.datetime,
    day_risk: DayRisk,
    min_seconds_between_entries: int,
    hard_caps: List[str],
    post_flat_cooldown_sec: int,
    last_trade_close_ts: Optional[float],
    # Optional extras (safe defaults; only active if DayRisk has fields / caller passes values)
    post_loss_cooldown_sec: int = 0,
    max_trades_per_hour: int = 0,
) -> tuple[Optional[str], List[str], str]:
    """
    Compute the entry gate / caps.

    Returns:
        gate_reason: str or None
        caps:        list of cap strings
        idle_reason: string to show in heartbeat
    """
    now_ts = time.time()

    # ----------------------------
    # Base gate from DayRisk (day-loss, max trades, max consec losses, etc.)
    # ----------------------------
    gate_reason = day_risk.gate_reason(now_ct, min_seconds_between_entries)
    caps: List[str] = []

    # ----------------------------
    # Extra gate: post-flat cooldown after any closed trade
    # ----------------------------
    if post_flat_cooldown_sec > 0 and last_trade_close_ts is not None:
        try:
            since_close = now_ts - float(last_trade_close_ts)
            if since_close < float(post_flat_cooldown_sec):
                if gate_reason is None:
                    gate_reason = "post_flat_cooldown"
        except Exception:
            # Never let gating crash the loop
            pass

    # ----------------------------
    # NEW: post-loss cooldown (if DayRisk tracks last_loss_ts)
    # ----------------------------
    if post_loss_cooldown_sec and post_loss_cooldown_sec > 0:
        try:
            last_loss_ts = getattr(day_risk, "last_loss_ts", None)
            if last_loss_ts is not None:
                since_loss = now_ts - float(last_loss_ts)
                if since_loss < float(post_loss_cooldown_sec):
                    if gate_reason is None:
                        gate_reason = "post_loss_cooldown"
        except Exception:
            pass

    # ----------------------------
    # NEW: hourly overtrade cap (optional)
    # NOTE:
    # - trade_bridge increments trades_this_hour on *trade close*.
    # - If you want this cap to apply to *entries*, you should increment
    #   trades_this_hour on successful entry instead (entry path).
    # This still provides protection if your close rate is high, but entry-side
    # tracking is more correct.
    # ----------------------------
    if max_trades_per_hour and max_trades_per_hour > 0:
        try:
            hour_key = f"{now_ct.date().isoformat()}_{now_ct.hour:02d}"
            prev_key = getattr(day_risk, "hour_key", None)
            if prev_key != hour_key:
                setattr(day_risk, "hour_key", hour_key)
                setattr(day_risk, "trades_this_hour", 0)

            cur_hour = int(getattr(day_risk, "trades_this_hour", 0) or 0)
            if cur_hour >= int(max_trades_per_hour):
                if gate_reason is None:
                    gate_reason = "max_trades_per_hour"
        except Exception:
            pass

    # ----------------------------
    # Collect caps
    # ----------------------------
    if gate_reason is not None:
        caps.append(gate_reason)

    for c in hard_caps:
        if c and c not in caps:
            caps.append(c)

    # If we have hard caps but no gate_reason yet, use the first hard cap as reason
    if hard_caps and gate_reason is None:
        gate_reason = hard_caps[0]

    idle_reason = gate_reason or ""

    return gate_reason, caps, idle_reason
