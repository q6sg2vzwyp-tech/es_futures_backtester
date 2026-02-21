#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pt_runtime_state.py

Runtime state load/apply + build/save helpers for paper_trader.py.

Goals:
- Shrink paper_trader by centralizing verbose restore/save logic.
- Preserve best-effort semantics: never crash the bot due to state.
- Keep bars restore/save format stable:
  - bars: [{"ts": "...iso...", "close": float}, ...]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass
class RuntimeRestore:
    # session/date
    day_date: Optional[Any] = None  # dt.date (kept Any to avoid importing datetime here)
    last_regime: Optional[str] = None

    # counters
    trades_today: Optional[int] = None
    running_pnl_today: Optional[float] = None
    wins_today: Optional[int] = None
    losses_today: Optional[int] = None

    # risk/meta/equity
    day_R: Optional[float] = None
    consec_losses: Optional[int] = None
    week_R: Optional[float] = None
    meta_ema_R: Optional[float] = None
    meta_n: Optional[int] = None
    equity: Optional[float] = None
    equity_hwm: Optional[float] = None
    last_acct_netliq: Optional[float] = None

    # bars
    bars_restored: int = 0


def _safe_int(v: Any) -> Optional[int]:
    try:
        if v is None:
            return None
        return int(v)
    except Exception:
        return None


def _safe_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def restore_runtime_state(
    *,
    runtime_state: Optional[Dict[str, Any]],
    dt_mod,  # pass in datetime module (import datetime as dt in caller)
    bars,    # BarBuffer
    logger,
) -> RuntimeRestore:
    """
    Applies runtime_state into BarBuffer and returns values for caller to apply
    into local variables (day_risk, week_state, meta, equity, etc.).

    This function does NOT mutate day_risk/week_state/meta directly.
    Caller keeps control of which objects to update.
    """
    out = RuntimeRestore()

    if not runtime_state or not isinstance(runtime_state, dict):
        return out

    try:
        # day_date (string -> dt.date)
        day_str = runtime_state.get("day_date")
        if isinstance(day_str, str):
            try:
                out.day_date = dt_mod.date.fromisoformat(day_str)
            except Exception:
                out.day_date = None

        # counters
        out.trades_today = _safe_int(runtime_state.get("trades_today"))
        out.running_pnl_today = _safe_float(runtime_state.get("running_pnl_today"))
        out.wins_today = _safe_int(runtime_state.get("wins_today"))
        out.losses_today = _safe_int(runtime_state.get("losses_today"))

        # risk/meta/equity
        out.day_R = _safe_float(runtime_state.get("day_R"))
        out.consec_losses = _safe_int(runtime_state.get("consec_losses"))
        out.week_R = _safe_float(runtime_state.get("week_R"))
        out.meta_ema_R = _safe_float(runtime_state.get("meta_ema_R"))
        out.meta_n = _safe_int(runtime_state.get("meta_n"))
        out.equity = _safe_float(runtime_state.get("equity"))
        out.equity_hwm = _safe_float(runtime_state.get("equity_hwm"))
        out.last_acct_netliq = _safe_float(runtime_state.get("last_acct_netliq"))

        reg_val = runtime_state.get("last_regime")
        if isinstance(reg_val, str) and reg_val:
            out.last_regime = reg_val

        # bars restore
        bars_state = runtime_state.get("bars")
        restored = 0
        if isinstance(bars_state, list):
            for row in bars_state:
                if not isinstance(row, dict):
                    continue
                ts_str = row.get("ts")
                close_val = row.get("close")
                if ts_str is None or close_val is None:
                    continue
                try:
                    ts_obj = dt_mod.datetime.fromisoformat(str(ts_str))
                    close_f = float(close_val)
                except Exception:
                    continue
                try:
                    bars.add(ts_obj, close_f)
                    restored += 1
                except Exception:
                    continue
        out.bars_restored = restored

        logger.info("[pt_runtime_state] restored: bars=%d", restored)
    except Exception as e:
        logger.error("[pt_runtime_state] restore failed: %s", e)

    return out


def apply_restore_into_objects(
    *,
    restore: RuntimeRestore,
    day_risk,
    week_state,
    meta,
    logger,
) -> None:
    """
    Mutates day_risk/week_state/meta based on restore values when present.
    Mirrors the prior best-effort behavior.
    """
    try:
        if restore.day_R is not None and hasattr(day_risk, "day_R"):
            try:
                day_risk.day_R = float(restore.day_R)
            except Exception:
                pass

        if restore.consec_losses is not None and hasattr(day_risk, "consec_losses"):
            try:
                day_risk.consec_losses = int(restore.consec_losses)
            except Exception:
                pass

        if restore.week_R is not None and hasattr(week_state, "week_R"):
            try:
                week_state.week_R = float(restore.week_R)
            except Exception:
                pass

        if restore.meta_ema_R is not None and hasattr(meta, "ema_R"):
            try:
                meta.ema_R = float(restore.meta_ema_R)
            except Exception:
                pass

        if restore.meta_n is not None:
            # support both meta.n_trades and meta.n
            if hasattr(meta, "n_trades"):
                try:
                    meta.n_trades = int(restore.meta_n)
                except Exception:
                    pass
            elif hasattr(meta, "n"):
                try:
                    meta.n = int(restore.meta_n)
                except Exception:
                    pass

    except Exception as e:
        logger.error("[pt_runtime_state] apply_into_objects failed: %s", e)


def build_runtime_state_out(
    *,
    day_date,
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
    bars,  # BarBuffer
    max_bars_to_save: int,
) -> Dict[str, Any]:
    """
    Builds the runtime_state dict with bar payload capped to max_bars_to_save.
    """
    # bars payload
    bars_payload = []
    try:
        num_bars = len(bars.ts)
        start_idx = max(0, num_bars - int(max_bars_to_save))
        for i in range(start_idx, num_bars):
            try:
                bars_payload.append(
                    {
                        "ts": bars.ts[i].isoformat(timespec="seconds"),
                        "close": float(bars.close[i]),
                    }
                )
            except Exception:
                continue
    except Exception:
        bars_payload = []

    meta_n = 0
    try:
        meta_n = int(getattr(meta, "n_trades", getattr(meta, "n", 0)) or 0)
    except Exception:
        meta_n = 0

    out = {
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
    return out
