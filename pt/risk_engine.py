from __future__ import annotations

import datetime as dt
import time
from typing import Optional


class DayRisk:
    """Intraday risk gate + counters.

    Mirrors the prior inline DayRisk in paper_trader.py.
    """

    def __init__(
        self,
        loss_cap_R: float,
        max_trades: int,
        max_consec_losses: int,
        post_flat_cooldown_sec: int,
    ):
        self.loss_cap_R = float(loss_cap_R)
        self.max_trades = int(max_trades)
        self.max_consec_losses = int(max_consec_losses)
        self.post_flat_cooldown_sec = int(max(0, post_flat_cooldown_sec))
        self.reset()

    def reset(self):
        self.day_R = 0.0
        self.trades = 0
        self.cool_until: Optional[dt.datetime] = None
        self.halted = False
        self.last_entry_time: Optional[float] = None
        self.consec_losses = 0
        self.last_flat_fill_ts: Optional[float] = None

    def post_flat_cooldown_remaining(self) -> Optional[float]:
        if not self.last_flat_fill_ts or self.post_flat_cooldown_sec <= 0:
            return None
        elapsed = time.time() - self.last_flat_fill_ts
        remaining = self.post_flat_cooldown_sec - elapsed
        return remaining if remaining > 0 else None

    def in_post_flat_cooldown(self) -> bool:
        return self.post_flat_cooldown_remaining() is not None

    def gate_reason(self, now: dt.datetime, min_gap_s: int) -> Optional[str]:
        if self.halted:
            return "halted"
        if self.cool_until and now < self.cool_until:
            return "cooldown"
        if self.trades >= self.max_trades:
            return "max_trades"
        if self.day_R <= -abs(self.loss_cap_R):
            return "dayR_cap"
        if self.consec_losses >= self.max_consec_losses:
            return "consec_losses"
        if self.in_post_flat_cooldown():
            return "post_flat_cooldown"
        if self.last_entry_time and (time.time() - self.last_entry_time) < max(0, min_gap_s):
            return "min_gap_between_entries"
        return None

    def can_trade(self, now: dt.datetime, min_gap_s: int) -> bool:
        return self.gate_reason(now, min_gap_s) is None
