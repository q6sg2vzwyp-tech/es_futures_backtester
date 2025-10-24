from dataclasses import dataclass
from datetime import date, datetime, timedelta


@dataclass
class BreakerState:
    daily_R: float = 0.0
    weekly_R: float = 0.0
    week_start: date | None = None
    tripped_today: bool = False


class CircuitBreakers:
    def __init__(self, daily_loss_R: float, weekly_loss_R: float, max_slippage_ticks: int):
        self.daily_limit = -abs(daily_loss_R)
        self.weekly_limit = -abs(weekly_loss_R)
        self.max_slip_ticks = int(max_slippage_ticks)
        self.state = BreakerState()

    def _week_of(self, d: date) -> date:
        return d - timedelta(days=d.weekday())

    def on_new_day(self, now: datetime):
        d = now.date()
        self.state.daily_R = 0.0
        self.state.tripped_today = False
        wk = self._week_of(d)
        if self.state.week_start is None or self.state.week_start != wk:
            self.state.week_start = wk
            self.state.weekly_R = 0.0

    def can_trade(self) -> bool:
        if self.state.tripped_today:
            return False
        if self.state.daily_R <= self.daily_limit:
            return False
        if self.state.weekly_R <= self.weekly_limit:
            return False
        return True

    def record_trade_R(self, r_value: float):
        self.state.daily_R += float(r_value)
        self.state.weekly_R += float(r_value)
        if self.state.daily_R <= self.daily_limit:
            self.state.tripped_today = True

    def slippage_ok(self, planned_ticks: float, actual_ticks: float) -> bool:
        return abs(actual_ticks - planned_ticks) <= self.max_slip_ticks

    def trip_today(self):
        self.state.tripped_today = True
