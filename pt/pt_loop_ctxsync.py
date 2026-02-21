#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pt_loop_ctxsync.py

End-of-iteration ctx synchronization.

Goal:
- Keep loop_core.py as a coordinator by centralizing the large ctx.update({...}) block.
- Behavior is intentionally identical to the inline update used in v8.

Design:
- loop_core passes locals() so we can pull the computed per-iteration variables without re-deriving them.
- eod_bayes_attempt_* fields are preserved from ctx (not locals), matching prior behavior.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping


def sync_ctx_end_of_iteration(ctx: Dict[str, Any], ns: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Update ctx with per-iteration state.

    Args:
        ctx: main loop context dict (mutated in-place)
        ns: a locals()-like mapping from loop_core for this iteration

    Returns:
        The updates dict that was applied (useful for tests / introspection).
    """

    # Most keys map 1:1 to same-named locals.
    direct_keys = [
        "BAYES_SOURCE",
        "AUTO_FLAT_CT",
        "day_date",
        "caps_reset_date",
        "bayes_ran_today",
        "safety_halt_for_today",
        "safety_last_ts",
        "equity",
        "equity_hwm",
        "hwm_factor",
        "last_acct_netliq",
        "trades_today",
        "total_trades",
        "running_pnl_today",
        "wins_today",
        "losses_today",
        "last_trade_close_ts",
        "last_acct_realized",
        "current_arm",
        "current_side",
        "in_real_window",
        "in_shadow_window",
        "gate_reason",
        "caps",
        "idle_reason",
        "state",
        "last_regime",
        "signal",
        "arm_score",
        "arm_score_aux",
        "regime",
        "adx_value",
        "atr_points",
        "last_ib_err",
        "LAST_ORPHAN_SWEEP_TS",
        "sharpe_R_value",
        "last_nostop_guard_ts",
        "last_state_save_ts",
        "pos_entry_ct",
        "pos_entry_px",
        "pos_entry_ts",
    ]

    updates: Dict[str, Any] = {}
    for k in direct_keys:
        # Preserve None if missing; loop_core previously had the key literal regardless.
        updates[k] = ns.get(k, None)

    # Explicit conversions that were inlined before.
    updates["shadow_enabled"] = bool(ns.get("shadow_enabled", False))

    try:
        updates["meta_factor"] = float(ns.get("meta_factor", 0.0) or 0.0)
    except Exception:
        updates["meta_factor"] = 0.0

    try:
        updates["boost_factor"] = float(ns.get("boost_factor", 0.0) or 0.0)
    except Exception:
        updates["boost_factor"] = 0.0

    # Preserve EOD Bayes latch fields from ctx (not locals), matching prior behavior.
    updates["eod_bayes_attempt_day"] = ctx.get("eod_bayes_attempt_day")

    try:
        updates["eod_bayes_attempt_ts"] = float(ctx.get("eod_bayes_attempt_ts", 0.0) or 0.0)
    except Exception:
        updates["eod_bayes_attempt_ts"] = 0.0

    ctx.update(updates)
    return updates
