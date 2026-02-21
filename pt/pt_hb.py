#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pt_hb.py

Centralized heartbeat assembly + write for paper_trader.py.

Goal:
- Remove duplicated hb_payload construction in multiple paths.
- Keep behavior stable (same fields, same merge rules).

Usage:
    from pt_hb import emit_hb
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from hb_core import build_heartbeat_payload
from pt_utils import hb_update_entry_and_unreal, normalize_es_avg_px, build_bandit_hb_fields


def emit_hb(
    *,
    # required context
    now_ct,
    hb_state: str,
    idle_reason: str,
    hb_pos_state: str,
    net: int,
    day_risk,
    week_state,
    last_px: float,
    bars_len: int,
    caps: List[str],
    last_ib_err: Optional[Dict[str, Any]],
    bayes_source: str,
    restart_ct_str: str,
    meta,
    meta_factor: float,
    boost_mode: str,
    boost_factor: float,
    sharpe_R: float,
    current_arm: Optional[str],
    current_side: Optional[str],
    last_signal_arm: Optional[str],
    last_signal_side: Optional[str],
    regime: str,
    equity: float,
    equity_hwm: float,
    hwm_factor: float,
    shadow_fields: Dict[str, Any],  # expects keys shadow_pnl_today, shadow_R_today, shadow_trades_today

    # market/order/pnl snapshots
    es_avg_px,
    es_unreal_pnl_raw,
    es_open_orders: int,
    es_open_stops: int,
    es_open_limits: int,
    open_order_ids,
    open_stop_ids,
    open_limit_ids,
    stop_px,
    target_px,
    acct_unreal_pnl,
    acct_realized_pnl,
    acct_netliq,

    # trade counters
    trades_today: int,
    total_trades: int,
    running_pnl_today: float,

    # dependencies
    bandit,
    margin_mgr,
    write_hb_fn,
    logger,
) -> Dict[str, Any]:
    """
    Builds HB payload, merges bandit + margin fields, writes via write_hb_fn, returns payload.
    """

    # Normalize avg px and compute locked entry/unreal fields for HB
    es_avg_px_norm = normalize_es_avg_px(es_avg_px, last_px, logger)
    entry_px_for_hb, unreal_for_hb = hb_update_entry_and_unreal(
        hb_pos_state=hb_pos_state,
        net=net,
        last_px=last_px,
        es_avg_px=es_avg_px_norm,
        es_unreal_pnl_raw=es_unreal_pnl_raw,
    )
    if unreal_for_hb is None:
        unreal_for_hb = 0.0

    hb_payload = build_heartbeat_payload(
        now_ct=now_ct,
        hb_state=hb_state,
        idle_reason=idle_reason,
        hb_pos_state=hb_pos_state,
        net=net,
        day_risk=day_risk,
        week_state=week_state,
        last_px=last_px,
        bars_len=bars_len,
        caps=caps,
        last_ib_err=last_ib_err,
        bayes_source=bayes_source,
        restart_ct_str=restart_ct_str,
        meta=meta,
        meta_factor=meta_factor,
        boost_mode=boost_mode,
        boost_factor=boost_factor,
        sharpe_R=sharpe_R,
        current_arm=current_arm,
        current_side=current_side,
        last_signal_arm=last_signal_arm,
        last_signal_side=last_signal_side,
        regime=regime,
        equity=equity,
        equity_hwm=equity_hwm,
        hwm_factor=hwm_factor,
        shadow_pnl_today=shadow_fields.get("shadow_pnl_today", 0.0),
        shadow_R_today=shadow_fields.get("shadow_R_today", 0.0),
        shadow_trades_today=shadow_fields.get("shadow_trades_today", 0),
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

    # Merge bandit fields
    try:
        hb_payload.update(build_bandit_hb_fields(bandit))
    except Exception as e:
        logger.error("[pt_hb] build_bandit_hb_fields failed: %s", e)

    # Merge margin fields
    try:
        hb_payload.update(margin_mgr.heartbeat_fields())
    except Exception as e:
        logger.error("[pt_hb] margin_mgr.heartbeat_fields failed: %s", e)

    # Write
    try:
        write_hb_fn(hb_payload)
    except Exception as e:
        logger.error("[pt_hb] write_hb_fn failed: %s", e)

    return hb_payload
