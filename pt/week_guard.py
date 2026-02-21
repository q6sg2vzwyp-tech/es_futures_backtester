from __future__ import annotations

import datetime as dt
from typing import Any, Dict, Optional, Callable


def iso_week_id(d: dt.date) -> str:
    y, w, _ = d.isocalendar()
    return f"{y}-W{w:02d}"


def weekly_cap_R(args) -> float:
    # cap is negative (loss limit) in R units
    return -float(getattr(args, "weekly_cap_mult", 0.0)) * abs(float(getattr(args, "day_loss_cap_R", 0.0)))


def init_week_state(
    restored_week_R: float,
    restored_week_id: str,
    today: dt.date,
    args,
) -> Dict[str, Any]:
    last_week = (restored_week_id or iso_week_id(today))
    return {
        "week_R": float(restored_week_R),
        "week_halted": False,
        "last_week_id": str(last_week),
        "weekly_cap_R": float(weekly_cap_R(args)),
    }


def weekly_reset_if_needed(
    today: dt.date,
    *,
    week_R: float,
    week_halted: bool,
    last_week_id: str,
    args,
    log: Optional[Callable[..., None]] = None,
) -> Dict[str, Any]:
    cur = iso_week_id(today)
    if cur != last_week_id:
        week_R = 0.0
        week_halted = False
        last_week_id = cur
        if log:
            try:
                log("weekly_reset", week_id=cur)
            except Exception:
                pass
    return {
        "week_R": float(week_R),
        "week_halted": bool(week_halted),
        "last_week_id": str(last_week_id),
        "weekly_cap_R": float(weekly_cap_R(args)),
    }


def add_week_reward(week_R: float, reward_R: float) -> float:
    return float(week_R) + float(reward_R)


def weekly_cap_hit(week_R: float, weekly_cap_R_val: float) -> bool:
    return float(week_R) <= float(weekly_cap_R_val)
