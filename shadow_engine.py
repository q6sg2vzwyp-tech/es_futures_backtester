#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
shadow_engine.py

Single-position virtual execution engine for shadow trading.

Design goals:
- Deterministic, dependency-free
- Supports: open, hold, close, flip
- Emits realized PnL on closes and flip-close legs
- Maintains avg_px / entry_px for current virtual position
- No I/O; caller handles logging (shadow_runner.py)

Signals:
- "LONG"  -> target net_qty = +qty
- "SHORT" -> target net_qty = -qty
- "FLAT"  -> target net_qty = 0
- ""/None -> NOOP (hold current position)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class ShadowEngineState:
    net_qty: int = 0                 # -qty, 0, +qty
    avg_px: float = 0.0              # average entry price for current position
    entry_px: float = 0.0            # alias for avg_px (kept for convenience)
    last_mark_px: float = 0.0        # last marked price
    last_reason: str = ""            # last reason string


def _sign(x: int) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)


def _target_from_signal(signal: str, qty: int) -> Optional[int]:
    s = (signal or "").upper().strip()
    if s == "":
        return None  # NOOP
    if s == "FLAT":
        return 0
    if s == "LONG":
        return int(abs(qty))
    if s == "SHORT":
        return -int(abs(qty))
    # Unknown signal -> treat as NOOP
    return None


def shadow_engine_step(
    st: ShadowEngineState,
    *,
    mark_px: float,
    signal: str,
    point_value: float = 50.0,   # ES: $50 per point per contract
    qty: int = 1,
    allow_flip: bool = True,
    reason: str = "",
) -> Dict[str, Any]:
    """
    Advance the engine by one step.

    Returns dict:
      net_qty, avg_px, entry_px,
      realized_pnl_usd (this step),
      prev_net_qty, prev_avg_px,
      action ("HOLD"|"OPEN"|"CLOSE"|"FLIP"|"NOOP"),
      reason, mark_px
    """
    px = float(mark_px)
    if px <= 0:
        # Defensive: invalid mark => NOOP
        return {
            "net_qty": int(st.net_qty),
            "avg_px": float(st.avg_px),
            "entry_px": float(st.entry_px),
            "realized_pnl_usd": 0.0,
            "prev_net_qty": int(st.net_qty),
            "prev_avg_px": float(st.avg_px),
            "action": "NOOP",
            "reason": "invalid_mark_px",
            "mark_px": float(st.last_mark_px or 0.0),
        }

    q = int(abs(qty)) if int(abs(qty)) > 0 else 1
    target = _target_from_signal(signal, q)

    prev_net = int(st.net_qty)
    prev_avg = float(st.avg_px)
    realized = 0.0
    action = "HOLD"

    st.last_mark_px = px
    st.last_reason = reason or ""

    # NOOP (hold)
    if target is None:
        return {
            "net_qty": int(st.net_qty),
            "avg_px": float(st.avg_px),
            "entry_px": float(st.entry_px),
            "realized_pnl_usd": 0.0,
            "prev_net_qty": prev_net,
            "prev_avg_px": prev_avg,
            "action": "NOOP",
            "reason": reason,
            "mark_px": px,
        }

    # If already at target: HOLD
    if prev_net == target:
        return {
            "net_qty": int(st.net_qty),
            "avg_px": float(st.avg_px),
            "entry_px": float(st.entry_px),
            "realized_pnl_usd": 0.0,
            "prev_net_qty": prev_net,
            "prev_avg_px": prev_avg,
            "action": "HOLD",
            "reason": reason,
            "mark_px": px,
        }

    # Flat -> Open
    if prev_net == 0 and target != 0:
        st.net_qty = int(target)
        st.avg_px = px
        st.entry_px = px
        action = "OPEN"
        return {
            "net_qty": int(st.net_qty),
            "avg_px": float(st.avg_px),
            "entry_px": float(st.entry_px),
            "realized_pnl_usd": 0.0,
            "prev_net_qty": prev_net,
            "prev_avg_px": prev_avg,
            "action": action,
            "reason": reason,
            "mark_px": px,
        }

    # Position -> Flat (close)
    if prev_net != 0 and target == 0:
        # realized pnl = (exit - entry) * point_value * net_qty
        # (net_qty sign handles long/short)
        realized = (px - prev_avg) * float(point_value) * float(prev_net)
        st.net_qty = 0
        st.avg_px = 0.0
        st.entry_px = 0.0
        action = "CLOSE"
        return {
            "net_qty": int(st.net_qty),
            "avg_px": float(st.avg_px),
            "entry_px": float(st.entry_px),
            "realized_pnl_usd": float(realized),
            "prev_net_qty": prev_net,
            "prev_avg_px": prev_avg,
            "action": action,
            "reason": reason,
            "mark_px": px,
            "prev_entry_px": prev_avg,
        }

    # Flip: long -> short OR short -> long
    if prev_net != 0 and target != 0 and _sign(prev_net) != _sign(target):
        if not allow_flip:
            # If flips disallowed, treat as HOLD
            return {
                "net_qty": int(st.net_qty),
                "avg_px": float(st.avg_px),
                "entry_px": float(st.entry_px),
                "realized_pnl_usd": 0.0,
                "prev_net_qty": prev_net,
                "prev_avg_px": prev_avg,
                "action": "HOLD",
                "reason": "flip_blocked",
                "mark_px": px,
            }

        # Step 1: close old
        realized = (px - prev_avg) * float(point_value) * float(prev_net)

        # Step 2: open new at current px
        st.net_qty = int(target)
        st.avg_px = px
        st.entry_px = px
        action = "FLIP"
        return {
            "net_qty": int(st.net_qty),
            "avg_px": float(st.avg_px),
            "entry_px": float(st.entry_px),
            "realized_pnl_usd": float(realized),
            "prev_net_qty": prev_net,
            "prev_avg_px": prev_avg,
            "action": action,
            "reason": reason,
            "mark_px": px,
            "prev_entry_px": prev_avg,
        }

    # Same-direction resize (not used in your current single-position design)
    # We keep it safe: snap to target and reset avg_px to current.
    if prev_net != 0 and target != 0 and _sign(prev_net) == _sign(target):
        st.net_qty = int(target)
        st.avg_px = prev_avg  # keep original avg
        st.entry_px = st.avg_px
        action = "HOLD"
        return {
            "net_qty": int(st.net_qty),
            "avg_px": float(st.avg_px),
            "entry_px": float(st.entry_px),
            "realized_pnl_usd": 0.0,
            "prev_net_qty": prev_net,
            "prev_avg_px": prev_avg,
            "action": action,
            "reason": "resize_no_realized",
            "mark_px": px,
        }

    # Fallback
    return {
        "net_qty": int(st.net_qty),
        "avg_px": float(st.avg_px),
        "entry_px": float(st.entry_px),
        "realized_pnl_usd": 0.0,
        "prev_net_qty": prev_net,
        "prev_avg_px": prev_avg,
        "action": "HOLD",
        "reason": "fallback",
        "mark_px": px,
    }
