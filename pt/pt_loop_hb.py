#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pt_loop_hb.py

Heartbeat helpers extracted from loop_core.py.

This module:
- Builds (extra_fields, shadow_fields) bundle
- Emits heartbeat via ctx['build_and_write_heartbeat'] (same as loop_core previously)

Best-effort: never raises.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import datetime as dt


def hb_fields(
    *,
    bandit: Any,
    shadow: Any,
    ctx: Dict[str, Any],
    build_bandit_hb_fields,
    shadow_enabled: bool,
    in_shadow_window: bool,
    in_real_window: bool,
    shadow_max_rts_day: int,
    shadow_max_rts_hour: int,
    shadow_post_close_cd: int,
    shadow_post_loss_cd: int,
    last_atr_points: float,
    last_adx_val: float,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    extra: Dict[str, Any] = {}
    try:
        extra.update(build_bandit_hb_fields(bandit) or {})
    except Exception:
        pass
    try:
        extra["real_arms"] = ctx.get("real_arms", [])
    except Exception:
        pass

    try:
        extra.update(
            {
                "shadow_roundtrips_csv": str(ctx.get("SHADOW_ROUNDTRIP_LOG", "") or ""),
                "shadow_enabled": bool(shadow_enabled),
                "atr_points": float(last_atr_points),
                "adx_val": float(last_adx_val),
                "in_shadow_window": bool(in_shadow_window),
                "in_real_window": bool(in_real_window),
                "shadow_max_roundtrips_per_day": int(shadow_max_rts_day),
                "shadow_max_roundtrips_per_hour": int(shadow_max_rts_hour),
                "shadow_post_close_cooldown_sec": int(shadow_post_close_cd),
                "shadow_post_loss_cooldown_sec": int(shadow_post_loss_cd),
            }
        )
    except Exception:
        pass

    sh: Dict[str, Any] = {}
    try:
        sh.update(shadow.heartbeat_fields() or {})
    except Exception:
        pass

    try:
        sh_last = ctx.get("shadow_last_status", {}) or {}
        if isinstance(sh_last, dict):
            sh.update(sh_last)
    except Exception:
        pass

    try:
        ctx["shadow_last_eval_ts"] = float(sh.get("shadow_last_eval_ts", 0.0) or 0.0)
        ctx["shadow_eval_count_today"] = int(sh.get("shadow_eval_count_today", 0) or 0)
    except Exception:
        pass

    return extra, sh


def emit(
    *,
    ctx: Dict[str, Any],
    now_ct: dt.datetime,
    hb_state: str,
    idle_reason: str,
    hb_pos_state: str,
    net: int,
    last_px: float,
    bars_len: int,
    caps: List[str],
    last_ib_err: Any,
    bayes_source: str,
    restart_ct_str: str,
    meta: Any,
    meta_factor: float,
    boost_mode: str,
    boost_factor: float,
    sharpe_R: float,
    current_arm: Any,
    current_side: Any,
    last_signal_arm: Any,
    last_signal_side: Any,
    regime: str,
    equity: float,
    equity_hwm: float,
    hwm_factor: float,
    trades_today: int,
    total_trades: int,
    running_pnl_today: float,
    shadow_fields: Dict[str, Any],
    extra_fields: Dict[str, Any],
    logger: Any,
) -> None:
    try:
        build_and_write_heartbeat = ctx["build_and_write_heartbeat"]
        build_and_write_heartbeat(
            ib=ctx["ib"],
            con=ctx["con"],
            hb_path=ctx["HB_PATH"],
            now_ct=now_ct,
            hb_state=hb_state,
            idle_reason=idle_reason,
            hb_pos_state=hb_pos_state,
            net=net,
            day_risk=ctx["day_risk"],
            week_state=ctx["week_state"],
            last_px=last_px,
            bars_len=bars_len,
            caps=caps,
            last_ib_err=last_ib_err,
            bayes_source=bayes_source,
            restart_ct_str=restart_ct_str,
            meta=meta,
            meta_factor=float(meta_factor),
            boost_mode=boost_mode,
            boost_factor=float(boost_factor),
            sharpe_R=float(sharpe_R),
            current_arm=current_arm,
            current_side=current_side,
            last_signal_arm=last_signal_arm,
            last_signal_side=last_signal_side,
            regime=str(regime or "unknown"),
            equity=float(equity),
            equity_hwm=float(equity_hwm),
            hwm_factor=float(hwm_factor),
            shadow_fields=shadow_fields,
            trades_today=int(trades_today),
            total_trades=int(total_trades),
            running_pnl_today=float(running_pnl_today),
            extra_fields=extra_fields,
            logger=logger,
        )
    except Exception:
        try:
            if logger is not None:
                logger.debug("[hb] emit failed (suppressed)")
        except Exception:
            pass
