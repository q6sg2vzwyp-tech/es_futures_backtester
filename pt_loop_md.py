#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pt_loop_md.py

Market-data phase extracted from loop_core.py.

Behavior is identical to the original block:
- reads last_price from ticker.last/marketPrice
- if None, emits heartbeat (md_no_last_price), sleeps, and returns early

This module is intentionally "thin": it does not decide windows or trading logic.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import time

import pt_loop_hb


def get_last_price_or_idle(
    *,
    ctx: Dict[str, Any],
    ticker: Any,
    now_ct,
    bandit: Any,
    shadow: Any,
    build_bandit_hb_fields,
    shadow_enabled: bool,
    shadow_max_rts_day: int,
    shadow_max_rts_hour: int,
    shadow_post_close_cd: int,
    shadow_post_loss_cd: int,
    last_atr_points: float,
    last_adx_val: float,
    last_ib_err: Any,
    bayes_source: str,
    daily_restart_ct_str: str,
    logger: Any,
) -> Tuple[Optional[float], bool]:
    """Returns (last_price, did_return_early)."""
    last_price = None
    try:
        last_price = ticker.last or ticker.marketPrice()
    except Exception:
        last_price = None

    if last_price is not None:
        try:
            return float(last_price), False
        except Exception:
            return None, False

    # Mirror the original loop_core behavior: heartbeat + sleep + return ctx.
    try:
        logger.warning("[md] no last price yet (ticker.last/marketPrice None); waiting...")
    except Exception:
        pass

    try:
        compute_position = ctx["compute_position"]
        ib = ctx["ib"]
        con = ctx["con"]
        net0 = int(round(compute_position(ib, con)))
    except Exception:
        net0 = 0

    hb_pos_state0 = "flat" if net0 == 0 else (f"long{net0}" if net0 > 0 else f"short{abs(net0)}")

    extra0, sh0 = pt_loop_hb.hb_fields(
        bandit=bandit,
        shadow=shadow,
        ctx=ctx,
        build_bandit_hb_fields=build_bandit_hb_fields,
        shadow_enabled=bool(shadow_enabled),
        in_shadow_window=False,
        in_real_window=False,
        shadow_max_rts_day=int(shadow_max_rts_day),
        shadow_max_rts_hour=int(shadow_max_rts_hour),
        shadow_post_close_cd=int(shadow_post_close_cd),
        shadow_post_loss_cd=int(shadow_post_loss_cd),
        last_atr_points=float(last_atr_points),
        last_adx_val=float(last_adx_val),
    )

    try:
        bars = ctx.get("bars", None)
        bars_len = int(bars.count()) if hasattr(bars, "count") else 0
    except Exception:
        bars_len = 0

    pt_loop_hb.emit(
        ctx=ctx,
        now_ct=now_ct,
        hb_state="idle",
        idle_reason="md_no_last_price",
        hb_pos_state=hb_pos_state0,
        net=net0,
        last_px=0.0,
        bars_len=int(bars_len),
        caps=["md_no_last_price"],
        last_ib_err=last_ib_err,
        bayes_source=bayes_source,
        restart_ct_str=str(daily_restart_ct_str or ""),
        meta=ctx.get("meta"),
        meta_factor=float(ctx.get("meta_factor", 1.0) or 1.0),
        boost_mode=getattr(ctx.get("args"), "boost_mode", "off"),
        boost_factor=float(ctx.get("boost_factor", 1.0) or 1.0),
        sharpe_R=float(ctx.get("sharpe_R_value", 0.0) or 0.0),
        current_arm=ctx.get("current_arm", None),
        current_side=ctx.get("current_side", None),
        last_signal_arm=ctx.get("last_signal_arm", None),
        last_signal_side=ctx.get("last_signal_side", None),
        regime=str(ctx.get("last_regime") or "unknown"),
        equity=float(ctx.get("equity", 100000.0) or 100000.0),
        equity_hwm=float(ctx.get("equity_hwm", ctx.get("equity", 100000.0)) or 100000.0),
        hwm_factor=float(ctx.get("hwm_factor", 1.0) or 1.0),
        trades_today=int(ctx.get("trades_today", 0) or 0),
        total_trades=int(ctx.get("total_trades", 0) or 0),
        running_pnl_today=float(ctx.get("running_pnl_today", 0.0) or 0.0),
        shadow_fields=sh0,
        extra_fields=extra0,
        logger=logger,
    )

    time.sleep(1.0)
    return None, True
