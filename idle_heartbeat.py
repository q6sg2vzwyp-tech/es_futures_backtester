#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
idle_heartbeat.py

Single emitter for all idle heartbeat states:
- weekend_lockout
- auto_flat
- outside_all_trading_windows
"""

from typing import List, Dict, Any
import time


def emit_idle_heartbeat(
    *,
    reason: str,
    now_ct,
    hb_pos_state,
    net,
    day_risk,
    week_state,
    last_px,
    bars,
    hard_caps,
    last_ib_err,
    BAYES_SOURCE,
    DAILY_RESTART_CT,
    meta,
    meta_factor,
    boost_mode,
    boost_factor,
    sharpe_R_value,
    current_arm,
    current_side,
    last_signal_arm,
    last_signal_side,
    last_regime,
    equity,
    equity_hwm,
    hwm_factor,
    sh,  # shadow heartbeat fields
    es_avg_px_norm,
    entry_px_for_hb,
    unreal_for_hb,
    es_open_orders,
    es_open_stops,
    es_open_limits,
    open_order_ids,
    open_stop_ids,
    open_limit_ids,
    stop_px,
    target_px,
    trades_today,
    total_trades,
    running_pnl_today,
    acct_unreal_pnl,
    acct_realized_pnl,
    acct_netliq,
    build_heartbeat_payload,
    write_hb,
    logger,
    sleep_sec: float = 1.0,
) -> None:
    """
    Emits a full heartbeat snapshot for idle states and sleeps.
    Caller should `continue` immediately after calling.
    """

    caps = [reason] + (hard_caps or [])

    hb_payload = build_heartbeat_payload(
        now_ct=now_ct,
        hb_state="idle",
        idle_reason=reason,
        hb_pos_state=hb_pos_state,
        net=net,
        day_risk=day_risk,
        week_state=week_state,
        last_px=last_px,
        bars_len=bars.count(),
        caps=caps,
        last_ib_err=last_ib_err,
        bayes_source=BAYES_SOURCE,
        restart_ct_str=DAILY_RESTART_CT.isoformat(timespec="minutes"),
        meta=meta,
        meta_factor=meta_factor,
        boost_mode=boost_mode,
        boost_factor=boost_factor,
        sharpe_R=sharpe_R_value,
        current_arm=current_arm,
        current_side=current_side,
        last_signal_arm=last_signal_arm,
        last_signal_side=last_signal_side,
        regime=last_regime,
        equity=equity,
        equity_hwm=equity_hwm,
        hwm_factor=hwm_factor,
        shadow_pnl_today=sh["shadow_pnl_today"],
        shadow_R_today=sh["shadow_R_today"],
        shadow_trades_today=sh["shadow_trades_today"],
        es_avg_px=es_avg_px_norm,
        entry_px_for_hb=entry_px_for_hb,
        unreal_for_hb=unreal_for_hb,
        es_open_orders=es_open_orders,
        es_open_stops=es_open_stops,
        es_open_limits=es_open_limits,
        open_order_ids=open_order_ids,
        open_stop_ids=open_stop_ids,
        open_limit_ids=open_limit_ids,
        stop_px=stop_px,
        target_px=target_px,
        trades_today=trades_today,
        total_trades=total_trades,
        running_pnl_today=running_pnl_today,
        acct_unreal_pnl=acct_unreal_pnl,
        acct_realized_pnl=acct_realized_pnl,
        acct_netliq=acct_netliq,
    )

    write_hb(hb_payload)
    time.sleep(sleep_sec)
