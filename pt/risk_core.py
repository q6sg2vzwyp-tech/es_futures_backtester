#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
risk_core.py

Day & week risk rails for ES Paper Trader.

Updates in this version:
- Adds optional fields used by newer gating + trade_bridge:
    * last_loss_ts       : wall-clock epoch seconds of last losing trade close
    * hour_key           : "YYYY-MM-DD_HH" key for current hour bucket
    * trades_this_hour   : count of closed trades in current hour bucket
- Extends gate_reason() with optional checks:
    * post_loss_cooldown_sec (if configured + last_loss_ts is known)
    * max_trades_per_hour (if configured + trades_this_hour is known)
  These are OFF by default (0) so behavior remains unchanged unless you enable them.

IMPORTANT FIXES (2026-01-02):
- Day R cap now works whether loss_cap_R is passed as +5.0 or -5.0 (CLI often uses -5.0).
- Weekly cap magnitude is computed from abs(day_loss_cap_R) so it is always positive.

Existing behavior remains intact for:
- max trades/day, max consec losses, post-flat cooldown, min time between entries
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Optional


@dataclass
class DayRisk:
    """
    Simple per-day risk manager.

    NOTE:
    - day_R is expressed in units of "R" if you pass R into register_closed_trade().
      If you pass R=None, day_R will not be updated and only trade counts / streaks
      will be enforced.
    """

    loss_cap_R: float
    max_trades: int
    max_consec_losses: int
    post_flat_cooldown_sec: int = 0

    # Optional gates (defaults keep existing behavior)
    post_loss_cooldown_sec: int = 0
    max_trades_per_hour: int = 0

    # State
    day_R: float = 0.0
    trades: int = 0
    consec_losses: int = 0

    last_flat_ts: Optional[dt.datetime] = None
    last_entry_time: Optional[float] = None  # wall-clock seconds since epoch

    # State used by trade_bridge + gate_core (safe defaults)
    last_loss_ts: Optional[float] = None      # epoch seconds (time.time())
    hour_key: Optional[str] = None            # "YYYY-MM-DD_HH"
    trades_this_hour: int = 0                 # closed trades in current hour

    def _ensure_hour_bucket(self, now: dt.datetime) -> None:
        """
        Keep trades_this_hour in the correct hourly bucket.
        """
        try:
            hk = f"{now.date().isoformat()}_{now.hour:02d}"
            if self.hour_key != hk:
                self.hour_key = hk
                self.trades_this_hour = 0
        except Exception:
            # If anything weird happens, do not break trading
            return

    def register_closed_trade(
        self,
        *,
        pnl_usd: float,
        R: Optional[float] = None,
        now: Optional[dt.datetime] = None,
    ) -> None:
        """
        Update intraday counters based on a closed trade.

        - pnl_usd: dollar PnL for the trade
        - R:      trade result in R units (optional; if None, day_R is unchanged)
        - now:    timestamp for bucketing (optional; defaults to dt.datetime.now())
        """
        self.trades += 1

        n = now or dt.datetime.now()
        self._ensure_hour_bucket(n)
        self.trades_this_hour += 1

        if pnl_usd < 0:
            self.consec_losses += 1
            try:
                import time
                self.last_loss_ts = float(time.time())
            except Exception:
                # preserve existing value if clock call fails
                self.last_loss_ts = self.last_loss_ts
        elif pnl_usd > 0:
            self.consec_losses = 0

        if R is not None:
            self.day_R += float(R)

    def mark_flat(self, now: dt.datetime) -> None:
        """
        Mark that we've gone flat (e.g., after a manual or safety flatten).
        Can be used to enforce a post-flat cooldown if desired.
        """
        self.last_flat_ts = now

    def gate_reason(self, now: dt.datetime, min_seconds_between_entries: int) -> Optional[str]:
        """
        Return a string reason why we should NOT allow a new entry right now,
        or None if entries are allowed.

        Reasons (if returned):
        - "day_R_cap"
        - "max_trades_per_day"
        - "max_consec_losses"
        - "post_flat_cooldown"
        - "post_loss_cooldown"
        - "max_trades_per_hour"
        - "min_seconds_between_entries"
        """
        # Day R cap: works for loss_cap_R passed as +5.0 or -5.0
        # (cap triggers once day_R <= -abs(loss_cap_R))
        if self.loss_cap_R != 0 and self.day_R <= -abs(self.loss_cap_R):
            return "day_R_cap"

        # Max trades per day
        if self.max_trades > 0 and self.trades >= self.max_trades:
            return "max_trades_per_day"

        # Max consecutive losses
        if self.max_consec_losses > 0 and self.consec_losses >= self.max_consec_losses:
            return "max_consec_losses"

        # Post-flat cooldown (optional; only enforced if last_flat_ts is set)
        if self.post_flat_cooldown_sec > 0 and self.last_flat_ts is not None:
            dt_sec = (now - self.last_flat_ts).total_seconds()
            if dt_sec < self.post_flat_cooldown_sec:
                return "post_flat_cooldown"

        # Post-loss cooldown (optional; only enforced if last_loss_ts is set)
        if self.post_loss_cooldown_sec > 0 and self.last_loss_ts is not None:
            try:
                import time
                dt_sec = time.time() - float(self.last_loss_ts)
                if dt_sec < float(self.post_loss_cooldown_sec):
                    return "post_loss_cooldown"
            except Exception:
                pass

        # Max trades per hour (optional)
        if self.max_trades_per_hour > 0:
            try:
                self._ensure_hour_bucket(now)
                if int(self.trades_this_hour) >= int(self.max_trades_per_hour):
                    return "max_trades_per_hour"
            except Exception:
                pass

        # Min time between entries
        if self.last_entry_time is not None and min_seconds_between_entries > 0:
            import time
            dt_sec = time.time() - self.last_entry_time
            if dt_sec < min_seconds_between_entries:
                return "min_seconds_between_entries"

        return None

    def reset_for_new_day(self) -> None:
        """
        Reset intraday risk counters and cooldowns for a fresh trading day.
        Does NOT touch configuration (loss_cap_R, max_trades, etc.).
        """
        self.day_R = 0.0
        self.trades = 0
        self.consec_losses = 0
        self.last_flat_ts = None
        self.last_entry_time = None

        # Reset loss/hour tracking
        self.last_loss_ts = None
        self.hour_key = None
        self.trades_this_hour = 0

        # If you ever add a 'halted_for_day' flag, clear it here:
        if hasattr(self, "halted_for_day"):
            setattr(self, "halted_for_day", False)


@dataclass
class WeekState:
    """
    Simple weekly R tracker.

    - week_start_date: when this week started
    - week_R:         cumulative R for the week
    - weekly_cap_R:   optional cap magnitude in R for the week (stored as positive)
    """
    week_start_date: dt.date
    week_R: float = 0.0
    weekly_cap_R: float = 0.0


def default_week_state(day_loss_cap_R: float, weekly_cap_mult: float) -> WeekState:
    """
    Build a default WeekState given a day R loss cap and a multiplier.

    Example:
        day_loss_cap_R = -5.0
        weekly_cap_mult = 4.0
        => weekly_cap_R = 20.0
    """
    today = dt.date.today()
    weekly_cap_R = abs(float(day_loss_cap_R)) * float(weekly_cap_mult)
    return WeekState(week_start_date=today, week_R=0.0, weekly_cap_R=weekly_cap_R)


def roll_week_if_needed(ws: WeekState) -> WeekState:
    """
    If we're 7+ days past the week_start_date, roll to a new week.

    NOTE:
    - This is intentionally simple and does not try to follow ISO week numbers.
    - It's perfectly fine for risk caps; if you want ISO behavior later, this
      is the place to change it.
    """
    today = dt.date.today()
    if (today - ws.week_start_date).days >= 7:
        return WeekState(
            week_start_date=today,
            week_R=0.0,
            weekly_cap_R=ws.weekly_cap_R,
        )
    return ws
