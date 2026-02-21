#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pt_loop_timing.py

Centralizes loop timing logic:
- orphan sweep cooldown gate
- sharpe update cadence
- runtime state save cadence
- IB error decay

Keeps paper_trader.py cleaner and reduces repeated time math.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any
import datetime as dt
import time


@dataclass
class LoopTimers:
    orphan_sweep_cooldown_sec: float = 30.0
    sharpe_update_sec: float = 5.0
    state_save_sec: float = 5.0
    ib_error_decay_sec: float = 120.0

    last_orphan_sweep_ts: float = 0.0
    last_sharpe_update_ts: float = 0.0
    last_state_save_ts: float = 0.0

    def should_orphan_sweep(self, *, now_ts: float, net: int) -> bool:
        if int(net) != 0:
            return False
        return (now_ts - float(self.last_orphan_sweep_ts)) >= float(self.orphan_sweep_cooldown_sec)

    def mark_orphan_swept(self, *, now_ts: float) -> None:
        self.last_orphan_sweep_ts = float(now_ts)

    def should_update_sharpe(self, *, now_ts: float) -> bool:
        return (now_ts - float(self.last_sharpe_update_ts)) >= float(self.sharpe_update_sec)

    def mark_sharpe_updated(self, *, now_ts: float) -> None:
        self.last_sharpe_update_ts = float(now_ts)

    def should_save_state(self, *, now_ts: float) -> bool:
        return (now_ts - float(self.last_state_save_ts)) >= float(self.state_save_sec)

    def mark_state_saved(self, *, now_ts: float) -> None:
        self.last_state_save_ts = float(now_ts)

    def maybe_decay_ib_error(self, last_ib_err: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not last_ib_err:
            return None
        try:
            ts_str = str(last_ib_err.get("ts", "1970-01-01T00:00:00"))
            err_ts = dt.datetime.fromisoformat(ts_str)
        except Exception:
            err_ts = dt.datetime.now()

        if (dt.datetime.now() - err_ts).total_seconds() > float(self.ib_error_decay_sec):
            return None
        return last_ib_err
