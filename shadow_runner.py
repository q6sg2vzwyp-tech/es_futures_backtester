#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
shadow_runner.py

Stateful "shadow trading" runner:
- Takes (arm, side, regime, meta/week context) + latest price
- Converts to a simple shadow signal: LONG/SHORT/FLAT/NOOP
- Feeds shadow_engine to simulate PnL + roundtrip logging

2025-12-22 update:
- Added anti-overtrading throttles so shadow data is usable:
  * decision_bucket_sec: only allow one decision per bucket
  * min_hold_sec: don't flip/close too quickly
  * atr_floor_ticks: ignore signals when ATR is too low
- Added fields into the roundtrip log to support better modeling.

2025-12-23 fix:
- shadow_runner_reset_day() now hard-resets ENGINE position (net/avg/entry) so
  shadow_net cannot "stick" across days.
- Decision bucketing now updates last_bucket_key even when a decision is blocked,
  preventing repeated attempts in the same bucket from slipping through.
- Resets keep throttles consistent and safe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable
import datetime as dt
import time

from shadow_engine import ShadowEngineState, shadow_engine_step

AppendRoundtripFn = Callable[[Dict[str, Any]], None]


@dataclass
class ShadowRunnerState:
    engine: ShadowEngineState
    pnl_today: float = 0.0
    R_today: float = 0.0
    trades_today: int = 0

    # anti-overtrading state
    last_bucket_key: int = -1
    last_signal: str = ""
    pos_open_ts: float = 0.0  # epoch seconds when shadow position opened (or flipped)
    last_roundtrip_close_ts: float = 0.0  # epoch seconds of last CLOSE (for min_roundtrip_gap)


def shadow_runner_reset_day(st: ShadowRunnerState) -> None:
    """
    Reset daily PnL counters AND hard-reset the engine so no position state
    can leak across day boundaries.
    """
    st.pnl_today = 0.0
    st.R_today = 0.0
    st.trades_today = 0

    st.last_bucket_key = -1
    st.last_signal = ""
    st.pos_open_ts = 0.0
    st.last_roundtrip_close_ts = 0.0

    # CRITICAL: reset engine position state (prevents shadow_net "sticking")
    try:
        st.engine.net_qty = 0
    except Exception:
        pass
    try:
        st.engine.avg_px = 0.0
    except Exception:
        pass
    try:
        st.engine.entry_px = 0.0
    except Exception:
        pass
    try:
        st.engine.realized_pnl_usd = 0.0
    except Exception:
        pass


def _side_to_signal(side: Optional[str]) -> str:
    s = (side or "").upper().strip()
    if s == "BUY":
        return "LONG"
    if s == "SELL":
        return "SHORT"
    return ""  # NOOP


def shadow_runner_step(
    st: ShadowRunnerState,
    *,
    now_ct: dt.datetime,
    last_px: float,
    in_shadow_window: bool,
    arm: Optional[str],
    side: Optional[str],  # "BUY"/"SELL"
    per_contract_init: float,
    last_regime: str,
    week_R: float,
    meta_ema_R: float,
    append_shadow_roundtrip_log: Optional[AppendRoundtripFn] = None,
    qty: int = 1,
    point_value: float = 50.0,
    force_flat_outside_window: bool = True,

    # data-quality throttles
    atr_points: float = 0.0,
    tick_size: float = 0.25,
    decision_bucket_sec: int = 30,
    min_hold_sec: int = 120,
    min_roundtrip_gap_sec: float = 10.0,
    atr_floor_ticks: float = 2.0,
) -> Dict[str, Any]:
    """
    Returns a dict of shadow status (net, avg_px, pnl_today, R_today, trades_today, etc.)
    """

    px = float(last_px or 0.0)
    if px <= 0:
        return {
            "ok": False,
            "reason": "invalid_last_px",
            "shadow_net": int(getattr(st.engine, "net_qty", 0)),
            "shadow_avg_px": float(getattr(st.engine, "avg_px", 0.0)),
            "shadow_pnl_today": float(st.pnl_today),
            "shadow_R_today": float(st.R_today),
            "shadow_trades_today": int(st.trades_today),
        }

    now_ts = time.time()

    # Outside window: optionally force FLAT so shadow positions do not leak
    if (not in_shadow_window) and force_flat_outside_window:
        shadow_sig = "FLAT"
        reason = "outside_shadow_window"
    else:
        shadow_sig = _side_to_signal(side)
        reason = "signal"

    # ATR floor: ignore entries/flip-flops when ATR is too low (still allow forced FLAT)
    atr_floor_points = float(atr_floor_ticks) * float(tick_size or 0.25)
    if shadow_sig in ("LONG", "SHORT") and float(atr_points or 0.0) > 0.0:
        if float(atr_points) < atr_floor_points:
            shadow_sig = ""  # NOOP
            reason = f"atr_floor_block atr={float(atr_points):.4f} < floor={atr_floor_points:.4f}"

    # Decision bucketing: at most one decision per bucket
    bucket_sec = int(decision_bucket_sec or 0)
    if bucket_sec > 0:
        bucket_key = int(now_ts // bucket_sec)

        # If we are still in same bucket and we're attempting a repeat entry signal, NOOP.
        # IMPORTANT: still record bucket_key so repeated attempts don't "slip" on stale key.
        if bucket_key == st.last_bucket_key:
            if shadow_sig in ("LONG", "SHORT") and shadow_sig == st.last_signal:
                shadow_sig = ""  # NOOP
                reason = f"bucket_throttle bucket={bucket_sec}s"
        else:
            st.last_bucket_key = bucket_key

    # Min hold: if we have a position open, avoid flipping/closing too quickly
    net_before = int(getattr(st.engine, "net_qty", 0))
    if net_before != 0 and min_hold_sec and int(min_hold_sec) > 0:
        held = 0.0
        if st.pos_open_ts and st.pos_open_ts > 0:
            held = max(0.0, now_ts - float(st.pos_open_ts))

        wants_change = False
        if shadow_sig == "FLAT":
            wants_change = True
        elif shadow_sig == "LONG" and net_before < 0:
            wants_change = True
        elif shadow_sig == "SHORT" and net_before > 0:
            wants_change = True

        if wants_change and (held < float(min_hold_sec)) and (reason != "outside_shadow_window"):
            shadow_sig = ""  # NOOP
            reason = f"min_hold_block held={held:.1f}s < min_hold={int(min_hold_sec)}s"

    # Min roundtrip gap: after a CLOSE, wait before allowing a new OPEN
    if net_before == 0 and min_roundtrip_gap_sec and float(min_roundtrip_gap_sec) > 0:
        if st.last_roundtrip_close_ts and st.last_roundtrip_close_ts > 0:
            gap = max(0.0, now_ts - float(st.last_roundtrip_close_ts))
            if shadow_sig in ("LONG", "SHORT") and gap < float(min_roundtrip_gap_sec):
                shadow_sig = ""  # NOOP
                reason = f"roundtrip_gap_block gap={gap:.1f}s < min_gap={float(min_roundtrip_gap_sec):.1f}s"

    # Run engine
    res = shadow_engine_step(
        st.engine,
        mark_px=px,
        signal=shadow_sig,
        point_value=float(point_value),
        qty=int(qty),
        allow_flip=True,
        reason=reason,
    )

    # Only update last_signal when a new directional signal is actually considered.
    # (Keep prior value when we NOOP so bucket logic remains meaningful.)
    if shadow_sig in ("LONG", "SHORT", "FLAT"):
        st.last_signal = shadow_sig

    realized = float(res.get("realized_pnl_usd", 0.0) or 0.0)
    action = str(res.get("action", "") or "")

    # Track pos open ts for min-hold
    if action in ("OPEN", "FLIP"):
        st.pos_open_ts = now_ts
    elif action == "CLOSE":
        st.pos_open_ts = 0.0
        st.last_roundtrip_close_ts = now_ts

    # Book realized into today totals
    if realized != 0.0:
        st.pnl_today += realized
        denom = float(per_contract_init or 1.0)
        R = realized / denom if denom != 0 else 0.0
        st.R_today += float(R)
        st.trades_today += 1

        if append_shadow_roundtrip_log is not None:
            try:
                row = {
                    "ts": now_ct.isoformat(timespec="seconds"),
                    "regime": str(last_regime or "unknown"),
                    "arm": str(arm or ""),
                    "side": str((side or "").upper()),
                    "action": action,
                    "mark_px": float(px),
                    "realized_pnl_usd": float(realized),
                    "R": float(R),
                    "week_R": float(week_R or 0.0),
                    "meta_ema_R": float(meta_ema_R or 0.0),
                    "atr_points": float(atr_points or 0.0),
                    "tick_size": float(tick_size or 0.25),
                    "decision_bucket_sec": int(decision_bucket_sec or 0),
                    "min_hold_sec": int(min_hold_sec or 0),
                    "atr_floor_ticks": float(atr_floor_ticks or 0.0),
                    "reason": str(reason or ""),
                }
                append_shadow_roundtrip_log(row)
            except Exception:
                pass

    return {
        "ok": True,
        "reason": reason,
        "shadow_net": int(getattr(st.engine, "net_qty", 0)),
        "shadow_avg_px": float(getattr(st.engine, "avg_px", 0.0)),
        "shadow_entry_px": float(getattr(st.engine, "entry_px", 0.0)),
        "shadow_pnl_today": float(st.pnl_today),
        "shadow_R_today": float(st.R_today),
        "shadow_trades_today": int(st.trades_today),
        "shadow_last_action": action,
        "shadow_last_realized": float(realized),
    }
