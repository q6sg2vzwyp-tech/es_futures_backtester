#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pt_loop_state.py

Extracted runtime-state persistence helpers from loop_core.py.

Goals:
- Keep loop_core.py as an orchestrator.
- Preserve existing runtime_state.json schema and save cadence behavior.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from state_core import save_runtime_state


def persist_runtime_state_throttled(
    *,
    args: Any,
    now_ts: float,
    last_state_save_ts: float,
    state_save_every_default: float,
    runtime_state_json: str,
    logger: Any,
    # payload fields
    day_date: Any,
    trades_today: int,
    consec_losses: int,
    bayes_ran_today: bool,
    equity: float,
    equity_hwm: float,
    last_acct_netliq: Optional[float],
    last_regime: str,
    bars: Any,
    pos_entry_px: Optional[float],
    pos_entry_ts: Optional[Any],
) -> float:
    """Persist runtime state to JSON at a throttled cadence.

    Returns updated last_state_save_ts (may be unchanged).
    """
    try:
        state_save_every = float(
            getattr(args, "state_save_every_sec", state_save_every_default) or state_save_every_default
        )
    except Exception:
        state_save_every = float(state_save_every_default)

    if state_save_every < 1.0:
        state_save_every = 1.0

    if (now_ts - float(last_state_save_ts or 0.0)) < state_save_every:
        return float(last_state_save_ts or 0.0)

    try:
        ts_list = list(getattr(bars, "ts", []) or [])
        close_list = list(getattr(bars, "close", []) or [])
        start_i = max(0, len(ts_list) - 256)

        bars_tail = []
        for i in range(start_i, len(ts_list)):
            try:
                ts_i = ts_list[i]
                close_i = close_list[i]
                bars_tail.append({"ts": str(ts_i), "close": float(close_i)})
            except Exception:
                continue

        runtime_state_out: Dict[str, Any] = {
            "day_date": day_date.isoformat() if hasattr(day_date, "isoformat") else str(day_date),
            "trades_today": int(trades_today),
            "consec_losses": int(consec_losses),
            "bayes_ran_today": bool(bayes_ran_today),
            "equity": float(equity),
            "equity_hwm": float(equity_hwm),
            "last_acct_netliq": float(last_acct_netliq) if last_acct_netliq is not None else None,
            "last_regime": str(last_regime or ""),
            "bars": bars_tail,
            "pos_entry_px": float(pos_entry_px) if pos_entry_px is not None else None,
            "pos_entry_ts": str(pos_entry_ts) if pos_entry_ts is not None else None,
        }

        save_runtime_state(runtime_state_json, runtime_state_out, logger=logger)
        return float(now_ts)

    except Exception as e:
        try:
            logger.error("[state_core] failed to save runtime state: %s", e)
        except Exception:
            pass
        return float(last_state_save_ts or 0.0)
