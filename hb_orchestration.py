# hb_orchestration.py
from __future__ import annotations

from typing import Any, Dict, List, Optional


def build_hb_payload_common(
    *,
    build_heartbeat_payload_fn,
    build_bandit_hb_fields_fn,
    margin_mgr,
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
    last_ib_err,
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
    shadow_fields: Dict[str, Any],
    es_avg_px: float,
    entry_px_for_hb: Optional[float],
    unreal_for_hb: float,
    es_open_orders: int,
    es_open_stops: int,
    es_open_limits: int,
    open_order_ids: List[int],
    open_stop_ids: List[int],
    open_limit_ids: List[int],
    stop_px: Optional[float],
    target_px: Optional[float],
    trades_today: int,
    total_trades: int,
    running_pnl_today: float,
    acct_unreal_pnl: float,
    acct_realized_pnl: float,
    acct_netliq: float,
    bandit_obj,
) -> Dict[str, Any]:
    payload = build_heartbeat_payload_fn(
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
        es_avg_px=es_avg_px,
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

    payload.update(build_bandit_hb_fields_fn(bandit_obj))

    try:
        payload.update(margin_mgr.heartbeat_fields())
    except Exception:
        # heartbeat must never crash
        pass

    return payload


def safe_write_hb(write_hb_fn, payload: Dict[str, Any]) -> None:
    try:
        write_hb_fn(payload)
    except Exception:
        return
