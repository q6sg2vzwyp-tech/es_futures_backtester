#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
hb_core.py

Heartbeat payload builder + writer for ES Paper Trader.

Goal:
- Keep the "HB schema" in one place (hb_monitor expects these fields).
- Let paper_trader.py call ONE function that:
  1) snapshots IB state (PnL + orders)
  2) normalizes avg_px / computes entry_px + unreal for display
  3) builds the HB payload
  4) merges optional extra fields (bandit + margin + any future)
  5) writes heartbeat atomically via hb_emit (with fallback)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import datetime as dt
import os
import json

from risk_core import DayRisk, WeekState
from learner_meta import MetaLearner

# Snapshot/normalize helpers
from pnl_core import snapshot_es_pnl_and_orders
from pt_utils import normalize_es_avg_px, hb_update_entry_and_unreal

try:
    from hb_emit import emit_hb_snapshot
except Exception:  # pragma: no cover
    emit_hb_snapshot = None  # type: ignore[assignment]


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _compute_drawdown_pct(equity: float, equity_hwm: float) -> float:
    """
    Compute drawdown as a positive percent from HWM:
      dd% = max(0, (HWM - equity) / HWM * 100)
    """
    try:
        hwm = float(equity_hwm)
        eq = float(equity)
        if hwm <= 0:
            return 0.0
        dd = (hwm - eq) / hwm * 100.0
        if dd < 0:
            dd = 0.0
        return float(dd)
    except Exception:
        return 0.0


def build_heartbeat_payload(
    *,
    now_ct: dt.datetime,
    hb_state: str,
    idle_reason: str,
    hb_pos_state: str,
    net: int,
    day_risk: DayRisk,
    week_state: WeekState,
    last_px: float,
    bars_len: int,
    caps: List[str],
    last_ib_err: Optional[Dict[str, Any]],
    bayes_source: str,
    restart_ct_str: str,
    meta: MetaLearner,
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
    shadow_pnl_today: float,
    shadow_R_today: float,
    shadow_trades_today: int,
    es_avg_px: Optional[float],
    entry_px_for_hb: Optional[float],
    unreal_for_hb: Optional[float],
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
    acct_unreal_pnl: Optional[float],
    acct_realized_pnl: Optional[float],
    acct_netliq: Optional[float],
) -> Dict[str, Any]:
    """
    Build the heartbeat payload dict (schema used by hb_monitor).
    Caller may add extra fields before writing (e.g., bandit_arms, margin_*).
    """
    day_R = float(getattr(day_risk, "day_R", 0.0))
    week_R = float(getattr(week_state, "week_R", 0.0))
    meta_ema_R = float(getattr(meta, "ema_R", 0.0))

    eq = _safe_float(equity, 0.0)
    hwm = _safe_float(equity_hwm, 0.0)
    dd_pct = _compute_drawdown_pct(eq, hwm)

    payload: Dict[str, Any] = {
        # Core status
        "ts": now_ct.isoformat(timespec="seconds"),
        "state": hb_state,
        "idle_reason": idle_reason,
        "pos_state": hb_pos_state,
        "net_qty": int(net),
        "day_R": float(day_R),
        "week_R": float(week_R),
        "px": float(last_px),
        "bars": int(bars_len),
        "caps": caps,
        "ib_err": last_ib_err,
        "bayes_source": bayes_source,
        "restart_ct": restart_ct_str,

        # Meta / learning / boost
        "meta_ema_R": float(meta_ema_R),
        "meta_aggr": float(meta_factor),
        "boost_mode": str(boost_mode),
        "boost_factor": float(boost_factor),
        "sharpe_R": float(sharpe_R),

        # Strategy
        "current_arm": current_arm,
        "current_side": current_side,
        "last_signal_arm": last_signal_arm,
        "last_signal_side": last_signal_side,
        "regime": regime,  # "trend" / "chop" / "unknown"

        # Equity / HWM
        "equity": float(eq),
        "equity_hwm": float(hwm),
        "hwm_factor": float(hwm_factor),
        "drawdown_pct": float(dd_pct),

        # Shadow learning (minimum fields)
        "shadow_pnl_today": float(shadow_pnl_today),
        "shadow_R_today": float(shadow_R_today),
        "shadow_trades_today": int(shadow_trades_today),

        # Price / PnL (display)
        "avg_px": es_avg_px,
        "entry_px": entry_px_for_hb,
        "pnl_unreal_usd": unreal_for_hb,

        # Orders
        "open_orders": int(es_open_orders),
        "open_stops": int(es_open_stops),
        "open_limits": int(es_open_limits),
        "open_order_ids": open_order_ids,
        "open_stop_ids": open_stop_ids,
        "open_limit_ids": open_limit_ids,
        "stop_px": stop_px,
        "target_px": target_px,

        # Counters / account
        "trades_today": int(trades_today),
        "total_trades": int(total_trades),
        "running_pnl_today": float(running_pnl_today),
        "acct_unreal_pnl": acct_unreal_pnl,
        "acct_realized_pnl": acct_realized_pnl,
        "acct_netliq": acct_netliq,
    }

    return payload


def _write_hb(payload: Dict[str, Any], hb_path: str) -> None:
    """
    Write heartbeat atomically.

    Prefers hb_emit.emit_hb_snapshot() if available; falls back to atomic JSON write.
    """
    try:
        os.makedirs(os.path.dirname(hb_path), exist_ok=True)
    except Exception:
        pass

    # Preferred writer
    if emit_hb_snapshot is not None:
        try:
            emit_hb_snapshot(payload, hb_path)  # type: ignore[misc]
            _write_health(payload, hb_path)
            return
        except Exception:
            pass

    # Fallback: atomic replace
    try:
        tmp_path = hb_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False))
        os.replace(tmp_path, hb_path)
    except Exception:
        return

    _write_health(payload, hb_path)


def _write_health(payload: Dict[str, Any], hb_path: str) -> None:
    """Write a lightweight health snapshot alongside heartbeat.

    hb_monitor reads run/health.json. This file is intended to be the
    most stable, low-risk health surface for watchdogs/monitors.

    Never raises.
    """
    try:
        run_dir = os.path.dirname(hb_path) or "."
        health_path = os.path.join(run_dir, "health.json")
        os.makedirs(os.path.dirname(health_path), exist_ok=True)

        now_iso = payload.get("now_ct") or payload.get("ts") or payload.get("timestamp") or ""
        if not isinstance(now_iso, str):
            now_iso = str(now_iso)

        # Minimal stable schema
        health = {
            "ts": now_iso,
            "hb_state": payload.get("hb_state", payload.get("state", "")),
            "idle_reason": payload.get("idle_reason", ""),
            "net": payload.get("net", 0),
            "last_px": payload.get("last_px", 0.0),
            "bars_len": payload.get("bars_len", 0),
            "ib_connected": bool(payload.get("ib_connected", False)),
            "can_trade": bool(payload.get("can_trade", False)),
            "caps": payload.get("caps", []),
        }

        tmp = health_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(json.dumps(health, ensure_ascii=False))
        os.replace(tmp, health_path)
    except Exception:
        return


def build_and_write_heartbeat(
    *,
    # IB deps
    ib,
    con,
    hb_path: str,

    # Core context
    now_ct: dt.datetime,
    hb_state: str,
    idle_reason: str,
    hb_pos_state: str,
    net: int,
    day_risk: DayRisk,
    week_state: WeekState,
    last_px: float,
    bars_len: int,
    caps: List[str],
    last_ib_err: Optional[Dict[str, Any]],
    bayes_source: str,
    restart_ct_str: str,

    # Learning / factors
    meta: MetaLearner,
    meta_factor: float,
    boost_mode: str,
    boost_factor: float,
    sharpe_R: float,
    current_arm: Optional[str],
    current_side: Optional[str],
    last_signal_arm: Optional[str],
    last_signal_side: Optional[str],
    regime: str,

    # Equity + shadow + counters
    equity: float,
    equity_hwm: float,
    hwm_factor: float,
    shadow_fields: Dict[str, Any],
    trades_today: int,
    total_trades: int,
    running_pnl_today: float,

    # Optional extra fields (bandit + margin + anything else)
    extra_fields: Optional[Dict[str, Any]] = None,

    # Logger (passed through to snapshot/normalize)
    logger=None,
) -> Dict[str, Any]:
    """
    One-call heartbeat: snapshot -> normalize -> payload -> merge extras -> write.
    Returns payload (useful for debugging/tests).
    """
    (
        es_avg_px,
        es_unreal_pnl_raw,
        es_open_orders,
        es_open_stops,
        es_open_limits,
        open_order_ids,
        open_stop_ids,
        open_limit_ids,
        stop_px,
        target_px,
        acct_unreal_pnl,
        acct_realized_pnl,
        acct_netliq,
    ) = snapshot_es_pnl_and_orders(ib=ib, con=con, last_px=last_px, logger=logger)

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
        shadow_pnl_today=float(shadow_fields.get("shadow_pnl_today", 0.0) or 0.0),
        shadow_R_today=float(shadow_fields.get("shadow_R_today", 0.0) or 0.0),
        shadow_trades_today=int(shadow_fields.get("shadow_trades_today", 0) or 0),
        es_avg_px=es_avg_px_norm,
        entry_px_for_hb=entry_px_for_hb,
        unreal_for_hb=unreal_for_hb,
        es_open_orders=int(es_open_orders),
        es_open_stops=int(es_open_stops),
        es_open_limits=int(es_open_limits),
        open_order_ids=list(open_order_ids or []),
        open_stop_ids=list(open_stop_ids or []),
        open_limit_ids=list(open_limit_ids or []),
        stop_px=stop_px,
        target_px=target_px,
        trades_today=int(trades_today),
        total_trades=int(total_trades),
        running_pnl_today=float(running_pnl_today),
        acct_unreal_pnl=acct_unreal_pnl,
        acct_realized_pnl=acct_realized_pnl,
        acct_netliq=acct_netliq,
    )

    # NEW: merge through any additional shadow_* fields (without breaking schema)
    try:
        for k, v in (shadow_fields or {}).items():
            if isinstance(k, str) and k.startswith("shadow_") and (k not in payload):
                payload[k] = v
    except Exception:
        pass

    if extra_fields:
        try:
            payload.update(extra_fields)
        except Exception:
            pass

    try:
        payload["ib_connected"] = bool(getattr(ib, "isConnected")())
    except Exception:
        payload["ib_connected"] = False
    try:
        payload["can_trade"] = str(payload.get("hb_state","")).lower() in ("run","live")
    except Exception:
        payload["can_trade"] = False

    _write_hb(payload, hb_path)
    return payload
