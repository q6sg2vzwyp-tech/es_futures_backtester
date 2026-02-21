#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pt_state.py

Centralized runtime state load/apply + save for paper_trader.py.

Goals:
- Keep behavior the same.
- Reduce line count and risk of indentation regressions.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, Optional, Tuple


def apply_runtime_state(
    *,
    runtime_state: Optional[Dict[str, Any]],
    day_date: dt.date,
    trades_today: int,
    running_pnl_today: float,
    wins_today: int,
    losses_today: int,
    day_risk,
    week_state,
    meta,
    equity: float,
    equity_hwm: float,
    last_acct_netliq: Optional[float],
    last_regime: str,
    bars,
    logger,
) -> Tuple[
    dt.date, int, float, int, int, float, float, Optional[float], str
]:
    """
    Applies runtime_state into provided objects/vars.
    Returns updated scalars:
        (day_date, trades_today, running_pnl_today, wins_today, losses_today,
         equity, equity_hwm, last_acct_netliq, last_regime)
    """

    if not runtime_state:
        return (
            day_date,
            trades_today,
            running_pnl_today,
            wins_today,
            losses_today,
            equity,
            equity_hwm,
            last_acct_netliq,
            last_regime,
        )

    try:
        # day_date
        day_str = runtime_state.get("day_date")
        if isinstance(day_str, str):
            try:
                day_date = dt.date.fromisoformat(day_str)
            except Exception:
                pass

        trades_today = int(runtime_state.get("trades_today", trades_today))
        running_pnl_today = float(runtime_state.get("running_pnl_today", running_pnl_today))
        wins_today = int(runtime_state.get("wins_today", wins_today))
        losses_today = int(runtime_state.get("losses_today", losses_today))

        # day_risk
        day_R_val = runtime_state.get("day_R", None)
        if day_R_val is not None and hasattr(day_risk, "day_R"):
            try:
                day_risk.day_R = float(day_R_val)
            except Exception:
                pass

        consec_val = runtime_state.get("consec_losses", None)
        if consec_val is not None and hasattr(day_risk, "consec_losses"):
            try:
                day_risk.consec_losses = int(consec_val)
            except Exception:
                pass

        # week_state
        week_R_val = runtime_state.get("week_R", None)
        if week_R_val is not None and hasattr(week_state, "week_R"):
            try:
                week_state.week_R = float(week_R_val)
            except Exception:
                pass

        # meta
        meta_ema = runtime_state.get("meta_ema_R", None)
        if meta_ema is not None and hasattr(meta, "ema_R"):
            try:
                meta.ema_R = float(meta_ema)
            except Exception:
                pass

        meta_n = runtime_state.get("meta_n", None)
        if meta_n is not None:
            if hasattr(meta, "n_trades"):
                try:
                    meta.n_trades = int(meta_n)
                except Exception:
                    pass
            elif hasattr(meta, "n"):
                try:
                    meta.n = int(meta_n)
                except Exception:
                    pass

        # equity
        eq_val = runtime_state.get("equity", None)
        if eq_val is not None:
            try:
                equity = float(eq_val)
            except Exception:
                pass

        hwm_val = runtime_state.get("equity_hwm", None)
        if hwm_val is not None:
            try:
                equity_hwm = float(hwm_val)
            except Exception:
                pass

        last_netliq_val = runtime_state.get("last_acct_netliq", None)
        if last_netliq_val is not None:
            try:
                last_acct_netliq = float(last_netliq_val)
            except Exception:
                pass

        reg_val = runtime_state.get("last_regime", None)
        if isinstance(reg_val, str) and reg_val:
            last_regime = reg_val

        # bars restore
        bars_state = runtime_state.get("bars", None)
        if isinstance(bars_state, list):
            restored = 0
            for row in bars_state:
                if not isinstance(row, dict):
                    continue
                ts_str = row.get("ts")
                close_val = row.get("close")
                if ts_str is None or close_val is None:
                    continue
                try:
                    ts_obj = dt.datetime.fromisoformat(str(ts_str))
                    close_f = float(close_val)
                except Exception:
                    continue
                try:
                    bars.add(ts_obj, close_f)
                    restored += 1
                except Exception:
                    continue
            logger.info("[pt_state] restored %d bars into BarBuffer", restored)

        logger.info("[pt_state] runtime state applied")
    except Exception as e:
        logger.error("[pt_state] failed to apply runtime state: %s", e)

    return (
        day_date,
        trades_today,
        running_pnl_today,
        wins_today,
        losses_today,
        equity,
        equity_hwm,
        last_acct_netliq,
        last_regime,
    )


def build_runtime_state_out(
    *,
    day_date: dt.date,
    trades_today: int,
    running_pnl_today: float,
    wins_today: int,
    losses_today: int,
    day_risk,
    week_state,
    meta,
    equity: float,
    equity_hwm: float,
    last_acct_netliq: Optional[float],
    last_regime: str,
    bars,
    max_bars_to_save: int = 256,
) -> Dict[str, Any]:
    """
    Returns the dict you should pass to save_runtime_state(...).
    """

    num_bars = len(getattr(bars, "ts", []))
    start_idx = max(0, num_bars - int(max_bars_to_save))

    bars_payload = []
    ts_list = getattr(bars, "ts", [])
    close_list = getattr(bars, "close", [])

    for i in range(start_idx, num_bars):
        try:
            bars_payload.append(
                {
                    "ts": ts_list[i].isoformat(timespec="seconds"),
                    "close": float(close_list[i]),
                }
            )
        except Exception:
            continue

    meta_n = int(getattr(meta, "n_trades", getattr(meta, "n", 0)) or 0)

    return {
        "day_date": day_date.isoformat(),
        "trades_today": int(trades_today),
        "running_pnl_today": float(running_pnl_today),
        "wins_today": int(wins_today),
        "losses_today": int(losses_today),
        "day_R": float(getattr(day_risk, "day_R", 0.0) or 0.0),
        "consec_losses": int(getattr(day_risk, "consec_losses", 0) or 0),
        "week_R": float(getattr(week_state, "week_R", 0.0) or 0.0),
        "meta_ema_R": float(getattr(meta, "ema_R", 0.0) or 0.0),
        "meta_n": meta_n,
        "equity": float(equity),
        "equity_hwm": float(equity_hwm),
        "last_acct_netliq": float(last_acct_netliq) if last_acct_netliq is not None else None,
        "last_regime": str(last_regime or "unknown"),
        "bars": bars_payload,
    }
