#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pt_hb_emit.py

Heartbeat build + emit helper to reduce duplication in paper_trader.py.

This module does NOT change strategy logic. It only centralizes:
- build_heartbeat_payload(...)
- adding bandit fields
- adding margin fields
- writing/emit to run/heartbeat.txt

Expected dependencies exist in your project:
- hb_core.build_heartbeat_payload
- pt_utils.build_bandit_hb_fields
- margin_core.MarginManager.heartbeat_fields (optional)
- hb_emit.emit_hb_snapshot (optional)
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

from hb_core import build_heartbeat_payload
from pt_utils import build_bandit_hb_fields


def _safe_emit(payload: Dict[str, Any], hb_path: str, emit_hb_snapshot=None) -> None:
    """
    Prefer hb_emit.emit_hb_snapshot(payload, hb_path) if provided.
    Otherwise write JSON to hb_path with a temp file swap.

    Windows note:
      os.replace() can transiently fail with WinError 5 ("Access is denied") if another
      process briefly touches heartbeat.txt. We retry a bit and then give up quietly.
      Heartbeat emission must NEVER crash the trading loop.
    """
    if emit_hb_snapshot is not None:
        try:
            emit_hb_snapshot(payload, hb_path)
            return
        except Exception:
            # fall back to local writer
            pass

    try:
        os.makedirs(os.path.dirname(hb_path), exist_ok=True)

        # Unique temp name to avoid collisions across multiple processes.
        # Use hidden-ish prefix to make it easy to clean up if needed.
        rand = str(int(time.time() * 1e6))
        tmp_path = hb_path + f".{rand}.tmp"

        # Write to temp, then atomically replace.
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_path, hb_path)
        return

    except Exception:
        return


def emit_heartbeat(
    *,
    hb_path: str,
    emit_hb_snapshot_fn,  # pass hb_emit.emit_hb_snapshot or None

    # core snapshot/time/state
    now_ct,
    hb_state: str,
    idle_reason: str,
    hb_pos_state: str,
    net: int,
    bars_len: int,
    caps: List[str],

    # model/risk state
    day_risk,
    week_state,
    meta,
    meta_factor: float,
    boost_mode: str,
    boost_factor: float,
    sharpe_R: float,

    # strategy/telemetry
    current_arm: Optional[str],
    current_side: Optional[str],
    last_signal_arm: Optional[str],
    last_signal_side: Optional[str],
    regime: str,

    # equity
    equity: float,
    equity_hwm: float,
    hwm_factor: float,

    # shadow fields
    shadow_pnl_today: float,
    shadow_R_today: float,
    shadow_trades_today: int,

    # market + position metrics
    last_px: float,
    es_avg_px: Optional[float],
    entry_px_for_hb: Optional[float],
    unreal_for_hb: float,

    # orders
    es_open_orders: int,
    es_open_stops: int,
    es_open_limits: int,
    open_order_ids: List[int],
    open_stop_ids: List[int],
    open_limit_ids: List[int],
    stop_px: Optional[float],
    target_px: Optional[float],

    # trade counters
    trades_today: int,
    total_trades: int,
    running_pnl_today: float,

    # account pnl
    acct_unreal_pnl: float,
    acct_realized_pnl: float,
    acct_netliq: Optional[float],

    # misc
    last_ib_err: Optional[Dict[str, Any]],
    bayes_source: str,
    restart_ct_str: str,

    # models/helpers to enrich heartbeat
    bandit,                 # Thompson model (for build_bandit_hb_fields)
    margin_mgr=None,         # MarginManager (optional)
) -> Dict[str, Any]:
    """
    Build a heartbeat payload, add bandit + margin fields, and write it.
    Returns the payload for logging/testing.
    """
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

    # Bandit fields
    try:
        payload.update(build_bandit_hb_fields(bandit))
    except Exception:
        pass

    # Margin fields
    if margin_mgr is not None:
        try:
            payload.update(margin_mgr.heartbeat_fields())
        except Exception:
            pass

    _safe_emit(payload, hb_path, emit_hb_snapshot=emit_hb_snapshot_fn)
    return payload

if __name__ == "__main__":
    from datetime import datetime, timezone

    now_local = datetime.now(timezone.utc).astimezone().isoformat()

    payload = {
        "ts": now_local,
        "state": "run",
        "idle_reason": "",
        "caps": [],
    }

    hb_path = os.path.join(os.path.dirname(__file__), "run", "heartbeat.txt")
    _safe_emit(payload, hb_path, emit_hb_snapshot=None)
