#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pt_loop_gates.py

Outside-window idle / return block extracted from loop_core.py.

This is the highest-value thinning win after v3 because it includes a large chunk of:
- force-path shadow stepping
- outside_all_trading_windows cap + heartbeat
- ctx persistence for end-of-iteration return
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import time


def maybe_return_outside_windows(
    *,
    ctx: Dict[str, Any],
    in_real_window: bool,
    in_shadow_window: bool,
    now_ct,
    last_px: float,
    net: int,
    per_contract_init: float,
    last_regime: str,
    hard_caps: List[str],
    logger: Any,
    hb_emit_fn,
    shadow_enabled: bool,
    # shadow rails and objects:
    shadow: Any,
    week_state: Any,
    meta: Any,
    args: Any,
    append_shadow_roundtrip_log,
    last_atr_points: float,
    shadow_decision_bucket_sec: int,
    shadow_min_hold_sec: int,
    shadow_max_hold_sec: Optional[int],
    shadow_max_rts_day: int,
    shadow_max_rts_hour: int,
    shadow_post_close_cd: int,
    shadow_post_loss_cd: int,
    # locals that must be persisted:
    pos_entry_ct,
    pos_entry_px,
    pos_entry_ts,
    current_arm,
    current_side,
    last_signal_arm,
    last_signal_side,
    total_trades: int,
    last_acct_realized,
    last_adx_val: float,
    # daily/guard state:
    day_date,
    caps_reset_date,
    bayes_ran_today: bool,
    safety_halt_for_today: bool,
    safety_last_ts,
    trades_today: int,
    running_pnl_today: float,
    wins_today: int,
    losses_today: int,
    last_trade_close_ts,
    eod_bayes_attempt_day,
    eod_bayes_attempt_ts: float,
) -> Optional[Dict[str, Any]]:
    if bool(in_real_window) or bool(in_shadow_window):
        return None

    # Keep original behavior: allow shadow to step on force path (outside both windows)
    try:
        sh_status = shadow.step(
            now_ct=now_ct,
            last_px=last_px,
            in_shadow_window=False,
            arm=None,
            side=None,
            per_contract_init=per_contract_init,
            last_regime=last_regime,
            week_R=float(getattr(week_state, "week_R", 0.0) or 0.0),
            meta_ema_R=float(getattr(meta, "ema_R", 0.0) or 0.0),
            append_shadow_roundtrip_log=append_shadow_roundtrip_log,
            atr_points=float(last_atr_points),
            tick_size=float(getattr(args, "tick_size", 0.25) or 0.25),
            decision_bucket_sec=int(shadow_decision_bucket_sec),
            min_hold_sec=int(shadow_min_hold_sec),
            atr_floor_ticks=2.0,
            max_hold_sec=shadow_max_hold_sec,
            shadow_enabled=bool(shadow_enabled),
            max_roundtrips_per_day=int(shadow_max_rts_day),
            max_roundtrips_per_hour=int(shadow_max_rts_hour),
            post_close_cooldown_sec=int(shadow_post_close_cd),
            post_loss_cooldown_sec=int(shadow_post_loss_cd),
        )
        ctx["shadow_last_status"] = dict(sh_status) if isinstance(sh_status, dict) else {}
    except Exception as e:
        try:
            logger.error("[shadow] step (force-flat) failed: %s", e)
        except Exception:
            pass

    if int(net) == 0:
        pos_entry_ct = None
        pos_entry_px = None
        pos_entry_ts = None

    caps_idle = ["outside_all_trading_windows"] + list(hard_caps or [])
    hb_emit_fn(hb_state="idle", idle_reason="outside_all_trading_windows", caps=caps_idle)

    time.sleep(1.0)
    ctx.update(
        {
            "day_date": day_date,
            "caps_reset_date": caps_reset_date,
            "bayes_ran_today": bayes_ran_today,
            "safety_halt_for_today": safety_halt_for_today,
            "safety_last_ts": safety_last_ts,
            "trades_today": trades_today,
            "running_pnl_today": running_pnl_today,
            "wins_today": wins_today,
            "losses_today": losses_today,
            "last_trade_close_ts": last_trade_close_ts,
            "pos_entry_ct": pos_entry_ct,
            "pos_entry_px": pos_entry_px,
            "pos_entry_ts": pos_entry_ts,
            "current_arm": current_arm,
            "current_side": current_side,
            "last_signal_arm": last_signal_arm,
            "last_signal_side": last_signal_side,
            "last_regime": last_regime,
            "total_trades": total_trades,
            "last_acct_realized": last_acct_realized,
            "last_atr_points": float(last_atr_points),
            "last_adx_val": float(last_adx_val),
            "shadow_enabled": bool(shadow_enabled),
            "eod_bayes_attempt_day": eod_bayes_attempt_day,
            "eod_bayes_attempt_ts": float(eod_bayes_attempt_ts or 0.0),
        }
    )
    return ctx
