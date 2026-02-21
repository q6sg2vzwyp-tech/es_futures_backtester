#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pt_hb_snapshot.py

Builds a heartbeat snapshot payload (dict) using injected callbacks.

PATCH (2026-01-04):
- Tolerate pnl=None by substituting PnlSnap.zero()
  (prevents AttributeError: 'NoneType' has no attribute 'es_avg_px')
- Call hb_update_entry_and_unreal in a kwargs-tolerant way.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pt_pnl_snap import PnlSnap


def emit_snapshot(
    *,
    now_ct,
    hb_state: str,
    idle_reason: str,
    hb_pos_state: str,
    net: int,
    day_risk,
    week_state,
    last_px: float,
    bars_len: int,
    caps,
    last_ib_err,
    bayes_source: str,
    restart_ct_str: str,
    meta,
    meta_factor: float,
    boost_mode: str,
    boost_factor: float,
    sharpe_R: float,
    current_arm,
    current_side,
    last_signal_arm,
    last_signal_side,
    regime: str,
    equity: float,
    equity_hwm: float,
    hwm_factor: float,
    shadow_pnl_today: float,
    shadow_R_today: float,
    shadow_trades_today: int,
    trades_today: int,
    total_trades: int,
    running_pnl_today: float,
    pnl: Optional[PnlSnap],
    bandit,
    build_heartbeat_payload,
    build_bandit_hb_fields,
    normalize_es_avg_px,
    hb_update_entry_and_unreal,
    write_hb,
    logger,
) -> Dict[str, Any]:
    """
    Returns the snapshot dict (and also writes it via write_hb()).

    IMPORTANT: This function is intentionally “plumbing-only”:
    it delegates formatting to build_heartbeat_payload and relies on injected helpers.
    """

    # ------------------------------------------------------------------
    # Hardening: allow smoke tests / callers to pass pnl=None.
    # ------------------------------------------------------------------
    if pnl is None:
        pnl = PnlSnap.zero()

    # Normalize avg px for display / consistency
    es_avg_px_norm = normalize_es_avg_px(pnl.es_avg_px, last_px, logger)

    # Update entry/unreal for HB display (tolerant to different helper signatures)
    entry_px_for_hb = None
    unreal_for_hb = None
    try:
        entry_px_for_hb, unreal_for_hb = hb_update_entry_and_unreal(
            hb_pos_state=hb_pos_state,
            net=net,
            last_px=last_px,
            es_avg_px=es_avg_px_norm,
            es_unreal_pnl_raw=pnl.es_unreal_pnl_raw,
        )
    except TypeError:
        # fallback if older helper uses positional signature
        try:
            entry_px_for_hb, unreal_for_hb = hb_update_entry_and_unreal(
                hb_pos_state, net, last_px, es_avg_px_norm, pnl.es_unreal_pnl_raw
            )
        except Exception:
            entry_px_for_hb, unreal_for_hb = (None, None)
    except Exception:
        entry_px_for_hb, unreal_for_hb = (None, None)

    # Build bandit fields (safe)
    bandit_fields: Dict[str, Any] = {}
    try:
        bandit_fields = dict(build_bandit_hb_fields(bandit) or {})
    except Exception:
        bandit_fields = {}

    # Delegate to the canonical payload builder (your existing hb schema)
    payload = build_heartbeat_payload(
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
        shadow_pnl_today=shadow_pnl_today,
        shadow_R_today=shadow_R_today,
        shadow_trades_today=shadow_trades_today,
        trades_today=trades_today,
        total_trades=total_trades,
        running_pnl_today=running_pnl_today,
        es_avg_px=es_avg_px_norm,
        entry_px_for_hb=entry_px_for_hb,
        unreal_for_hb=float(unreal_for_hb or 0.0),
        es_open_orders=int(pnl.es_open_orders or 0),
        es_open_stops=int(pnl.es_open_stops or 0),
        es_open_limits=int(pnl.es_open_limits or 0),
        open_order_ids=list(pnl.open_order_ids or []),
        open_stop_ids=list(pnl.open_stop_ids or []),
        open_limit_ids=list(pnl.open_limit_ids or []),
        stop_px=pnl.stop_px,
        target_px=pnl.target_px,
        acct_unreal_pnl=float(pnl.acct_unreal_pnl or 0.0),
        acct_realized_pnl=float(pnl.acct_realized_pnl or 0.0),
        acct_netliq=pnl.acct_netliq,
        bandit_fields=bandit_fields,
        logger=logger,
    )

    # Emit/write (caller controls how/where)
    try:
        write_hb(payload)
    except Exception:
        pass

    return payload
