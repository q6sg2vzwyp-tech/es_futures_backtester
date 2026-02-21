# runtime_state_core.py
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class RestoredState:
    day_date: dt.date
    trades_today: int
    running_pnl_today: float
    wins_today: int
    losses_today: int
    equity: float
    equity_hwm: float
    last_acct_netliq: Optional[float]
    last_regime: str


def apply_runtime_state(
    runtime_state: Optional[Dict[str, Any]],
    *,
    day_risk,
    week_state,
    meta,
    bars,
    logger,
    default_day_date: dt.date,
) -> Optional[RestoredState]:
    """
    Applies runtime_state dict onto in-memory objects (day_risk/week_state/meta/bars)
    and returns a compact RestoredState for the caller to use.
    """
    if not runtime_state:
        return None

    try:
        day_date = default_day_date
        day_str = runtime_state.get("day_date")
        if isinstance(day_str, str):
            try:
                day_date = dt.date.fromisoformat(day_str)
            except Exception:
                pass

        trades_today = int(runtime_state.get("trades_today", 0))
        running_pnl_today = float(runtime_state.get("running_pnl_today", 0.0))
        wins_today = int(runtime_state.get("wins_today", 0))
        losses_today = int(runtime_state.get("losses_today", 0))

        day_R_val = runtime_state.get("day_R", None)
        if day_R_val is not None:
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

        week_R_val = runtime_state.get("week_R", None)
        if week_R_val is not None and hasattr(week_state, "week_R"):
            try:
                week_state.week_R = float(week_R_val)
            except Exception:
                pass

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

        equity = float(runtime_state.get("equity", 100000.0) or 100000.0)
        equity_hwm = float(runtime_state.get("equity_hwm", equity) or equity)

        last_acct_netliq = runtime_state.get("last_acct_netliq", None)
        if last_acct_netliq is not None:
            try:
                last_acct_netliq = float(last_acct_netliq)
            except Exception:
                last_acct_netliq = None

        last_regime = runtime_state.get("last_regime", "unknown") or "unknown"

        # Restore bars if present
        bars_state = runtime_state.get("bars", None)
        if isinstance(bars_state, list):
            restored = 0
            for row in bars_state:
                try:
                    ts_str = row.get("ts")
                    close_val = row.get("close")
                    if not ts_str or close_val is None:
                        continue
                    ts_obj = dt.datetime.fromisoformat(ts_str)
                    close_f = float(close_val)
                    bars.add(ts_obj, close_f)
                    restored += 1
                except Exception:
                    continue
            try:
                logger.info("[runtime_state_core] restored %d bars into BarBuffer", restored)
            except Exception:
                pass

        return RestoredState(
            day_date=day_date,
            trades_today=trades_today,
            running_pnl_today=running_pnl_today,
            wins_today=wins_today,
            losses_today=losses_today,
            equity=equity,
            equity_hwm=equity_hwm,
            last_acct_netliq=last_acct_netliq,
            last_regime=str(last_regime),
        )

    except Exception as e:
        try:
            logger.error("[runtime_state_core] apply_runtime_state failed: %s", e)
        except Exception:
            pass
        return None
