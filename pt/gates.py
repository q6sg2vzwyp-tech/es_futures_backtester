from __future__ import annotations

from typing import Callable, List, Tuple, Any


def evaluate_gates(
    *,
    risk: Any,
    args: Any,
    day_realized: float,
    day_guard_dollars: float,
    week_R: float,
    weekly_cap_R: float,
    news_kill_active: bool,
    rt_fresh: bool,
    in_session_window: bool,
    week_halted: bool,
    weekly_cap_hit_fn: Callable[[float, float], bool],
) -> Tuple[str, List[str], bool]:
    """
    Pure gating evaluator.

    Returns:
      (state, caps_reasons, week_halted)

    State is one of: "caps", "wait_rt", "active", "sleep"
    """
    # Weekly cap gating (sticky until weekly reset logic clears it elsewhere)
    if weekly_cap_hit_fn(week_R, weekly_cap_R):
        week_halted = True

    # Peak DD guard + caps reasons
    caps_reasons: List[str] = []

    # Day loss cap (in R)
    try:
        day_R = float(getattr(risk, "day_R", 0.0))
    except Exception:
        day_R = 0.0
    if day_R <= -abs(float(getattr(args, "day_loss_cap_R", 0.0))):
        caps_reasons.append("dayR_cap")

    # Max trades
    try:
        trades = int(getattr(risk, "trades", 0))
    except Exception:
        trades = 0
    if trades >= int(getattr(args, "max_trades_per_day", 0)):
        caps_reasons.append("max_trades")

    # Explicit risk halt
    if bool(getattr(risk, "halted", False)):
        caps_reasons.append("risk_halted")

    # only check minus_pct_guard if day_guard_pct > 0
    try:
        day_guard_pct = float(getattr(args, "day_guard_pct", 0.0))
    except Exception:
        day_guard_pct = 0.0
    if day_guard_pct > 0 and float(day_realized) <= float(day_guard_dollars):
        caps_reasons.append("minus_pct_guard")

    # Weekly cap
    if weekly_cap_hit_fn(week_R, weekly_cap_R):
        caps_reasons.append("weekly_cap_R")

    # News kill
    if bool(news_kill_active):
        caps_reasons.append("news_kill")

    # Determine base state (before weekend/TOD overrides)
    if caps_reasons:
        state = "caps"
    elif bool(getattr(args, "require_rt_before_trading", False)) and (not bool(rt_fresh)):
        state = "wait_rt"
    else:
        state = "active" if bool(in_session_window) else "sleep"

    return state, caps_reasons, week_halted
