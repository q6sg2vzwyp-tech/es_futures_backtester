#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
session_core.py

Session / calendar helpers for ES Paper Trader.

Right now this module handles:
- Daily date roll reset (midnight boundary)
- Daily cap reset at a fixed AM session time (default 08:30 CT)

It keeps paper_trader.py slimmer by moving the mechanical
"reset counters/flags" logic into reusable helpers.
"""

from __future__ import annotations

from typing import Optional, Tuple
import datetime as dt

from risk_core import DayRisk


def reset_daily_flags(
    now_ct: dt.datetime,
    day_date: dt.date,
    bayes_ran_today: bool,
    friday_flat_done: bool,
    friday_flat_date: Optional[dt.date],
    holiday_flat_done: bool,
    holiday_flat_date: Optional[dt.date],
    safety_halt_for_today: bool,
    safety_last_ts: Optional[dt.datetime],
    trades_today: int,
    running_pnl_today: float,
    wins_today: int,
    losses_today: int,
    last_trade_close_ts: Optional[float],
) -> Tuple[
    dt.date,
    bool,
    bool,
    Optional[dt.date],
    bool,
    Optional[dt.date],
    bool,
    Optional[dt.datetime],
    int,
    float,
    int,
    int,
    Optional[float],
]:
    """
    Handle date roll at midnight.

    If the calendar date changed, this:
    - Resets daily counters
    - Clears "ran today" / flat-lock flags
    - Clears safety halt and last trade close timestamp

    Returns the updated values (same order as inputs, plus new date).
    """
    if now_ct.date() == day_date:
        # No change; pass everything through untouched
        return (
            day_date,
            bayes_ran_today,
            friday_flat_done,
            friday_flat_date,
            holiday_flat_done,
            holiday_flat_date,
            safety_halt_for_today,
            safety_last_ts,
            trades_today,
            running_pnl_today,
            wins_today,
            losses_today,
            last_trade_close_ts,
        )

    # New day: reset daily state
    new_day = now_ct.date()
    bayes_ran_today = False

    friday_flat_done = False
    friday_flat_date = None

    holiday_flat_done = False
    holiday_flat_date = None

    safety_halt_for_today = False
    safety_last_ts = None

    trades_today = 0
    running_pnl_today = 0.0
    wins_today = 0
    losses_today = 0
    last_trade_close_ts = None

    return (
        new_day,
        bayes_ran_today,
        friday_flat_done,
        friday_flat_date,
        holiday_flat_done,
        holiday_flat_date,
        safety_halt_for_today,
        safety_last_ts,
        trades_today,
        running_pnl_today,
        wins_today,
        losses_today,
        last_trade_close_ts,
    )


def reset_caps_for_new_session(
    now_ct: dt.datetime,
    caps_reset_date: Optional[dt.date],
    day_risk: DayRisk,
    safety_halt_for_today: bool,
    safety_last_ts: Optional[dt.datetime],
    trades_today: int,
    running_pnl_today: float,
    wins_today: int,
    losses_today: int,
    last_trade_close_ts: Optional[float],
    logger,
    caps_reset_time: dt.time = dt.time(8, 30),
) -> Tuple[
    Optional[dt.date],
    int,
    float,
    int,
    int,
    bool,
    Optional[dt.datetime],
    Optional[float],
]:
    """
    Reset intraday risk / counters at the AM session open (default 08:30 CT).
    """
    now_date = now_ct.date()
    now_time = now_ct.time()

    # ---- Trading day guard (weekends) ------------------------------------
    # ES should not reset session caps on Sat/Sun
    is_trading_day = now_ct.weekday() < 5

    # Helper: detect whether we already have "live" intraday stats
    has_intraday_stats = (
        trades_today != 0
        or abs(running_pnl_today) > 1e-9
        or wins_today != 0
        or losses_today != 0
    )

    # -------- First call after process startup: RESUME vs FRESH DAY --------
    if caps_reset_date is None:
        if has_intraday_stats:
            # We likely just rebuilt from trades.csv and want to RESUME the day.
            logger.info(
                "[caps_reset] first run for %s; preserving intraday stats (resume mode)",
                now_date.isoformat(),
            )
            caps_reset_date = now_date
            return (
                caps_reset_date,
                trades_today,
                running_pnl_today,
                wins_today,
                losses_today,
                safety_halt_for_today,
                safety_last_ts,
                last_trade_close_ts,
            )
        # Otherwise fall through and wait for a real trading session reset

    # -------- True "new session" reset (date change + after reset time) ----
    if (
        is_trading_day
        and (caps_reset_date is None or caps_reset_date != now_date)
        and now_time >= caps_reset_time
    ):
        logger.info(
            "[caps_reset] resetting day risk & safety flags for new session "
            "(date=%s, time=%s)",
            now_date,
            now_time,
        )
        caps_reset_date = now_date

        # Reset intraday counters
        trades_today = 0
        running_pnl_today = 0.0
        wins_today = 0
        losses_today = 0

        # Reset day risk rails
        try:
            day_risk.reset_for_new_day()
        except Exception as e:
            logger.error(f"[caps_reset] failed to reset DayRisk: {e}")

        # Clear safety halt so we can trade again
        safety_halt_for_today = False
        safety_last_ts = None
        last_trade_close_ts = None

    # Otherwise (weekend, same day, or before caps_reset_time), do nothing.
    return (
        caps_reset_date,
        trades_today,
        running_pnl_today,
        wins_today,
        losses_today,
        safety_halt_for_today,
        safety_last_ts,
        last_trade_close_ts,
    )
