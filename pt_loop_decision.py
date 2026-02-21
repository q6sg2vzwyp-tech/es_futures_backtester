#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pt_loop_decision.py

Entry eligibility / caps assembly / safety nostop guard extracted from loop_core.py.

This module centralizes:
- hard_caps assembly (safety_halt + policy hard_caps)
- compute_gate() call (min_seconds_between_entries + post-flat cooldown)
- day_risk hard gate_reason integration
- can_enter calculation
- safety nostop guard that attaches protection if a position exists with no stops/limits
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import time

import order_core


def evaluate_entry_eligibility(
    *,
    ctx: Dict[str, Any],
    args: Any,
    now_ct,
    in_real_window: bool,
    net: int,
    es_open_orders: int,
    es_open_stops: int,
    es_open_limits: int,
    last_fill_ts: Optional[float],
    last_trade_close_ts: Optional[float],
    day_risk: Any,
    policy_res: Any,
    safety_halt_for_today: bool,
    compute_gate,
    ib: Any,
    con: Any,
    last_px: float,
    logger: Any,
) -> Dict[str, Any]:
    # hard caps
    hard_caps: List[str] = []
    if bool(safety_halt_for_today):
        hard_caps.append("safety_halt_for_today")
    if policy_res is not None:
        try:
            hard_caps.extend(list(getattr(policy_res, "hard_caps", []) or []))
        except Exception:
            pass

    post_flat_cd = int(getattr(args, "post_flat_cooldown_sec", 0) or 0)

    gate_reason, caps, idle_reason = compute_gate(
        now_ct=now_ct,
        day_risk=day_risk,
        min_seconds_between_entries=int(getattr(args, "min_seconds_between_entries", 0) or 0),
        hard_caps=hard_caps,
        post_flat_cooldown_sec=post_flat_cd,
        last_trade_close_ts=last_trade_close_ts,
    )

    # HARD DAY-RISK ENTRY GATE (cannot be bypassed)
    try:
        dr_gr = day_risk.gate_reason()
    except Exception:
        dr_gr = None

    if dr_gr:
        gate_reason = gate_reason or str(dr_gr)
        if caps is None:
            caps = []
        if str(dr_gr) not in caps:
            caps = list(caps) + [str(dr_gr)]
        try:
            logger.warning(
                "[DAY_RISK_BLOCK] gate=%s day_R=%.3f cap=%.3f trades=%s consec_losses=%s",
                str(dr_gr),
                float(getattr(day_risk, "day_R", 0.0) or 0.0),
                float(getattr(day_risk, "loss_cap_R", 0.0) or 0.0),
                int(getattr(day_risk, "trades", 0) or 0),
                int(getattr(day_risk, "consec_losses", 0) or 0),
            )
        except Exception:
            pass

    can_enter = (
        gate_reason is None
        and bool(in_real_window)
        and int(net) == 0
        and int(es_open_orders) == 0
        and int(es_open_stops) == 0
        and int(es_open_limits) == 0
    )

    # safety nostop guard
    last_nostop_guard_ts = float(ctx.get("last_nostop_guard_ts", 0.0) or 0.0)
    safety_grace_sec = 1.0
    recently_filled = (last_fill_ts is not None) and ((time.time() - float(last_fill_ts)) < safety_grace_sec)
    nostop_cooldown_sec = 30.0

    if (
        bool(getattr(args, "place_orders", False))
        and int(net) != 0
        and int(es_open_stops) == 0
        and int(es_open_limits) == 0
        and (not recently_filled)
        and ((time.time() - float(last_nostop_guard_ts)) >= nostop_cooldown_sec)
    ):
        logger.warning("[safety_nostop] net position detected with NO protective STOP/TARGET; attaching protection")
        last_nostop_guard_ts = time.time()
        ctx["last_nostop_guard_ts"] = last_nostop_guard_ts
        try:
            order_core.guard_naked_position(ib=ib, contract=con, net_qty=int(net), last_px=float(last_px), args=args, logger=logger)
        except Exception as e:
            logger.error("[safety_nostop] failed to attach protection: %s", e)

    return {
        "hard_caps": hard_caps,
        "gate_reason": gate_reason,
        "caps": caps,
        "idle_reason": idle_reason,
        "can_enter": bool(can_enter),
        "last_nostop_guard_ts": float(last_nostop_guard_ts),
    }
