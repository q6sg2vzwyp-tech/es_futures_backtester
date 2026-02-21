from __future__ import annotations

# pt/utils_math.py
#
# Centralized small math helpers extracted from paper_trader.py.
# Side-effect free.

from typing import Union


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def ticks_to_price_delta(ticks: int, tick_size: float) -> float:
    return float(ticks) * float(tick_size)


def round_to_tick(p: float, tick: float) -> float:
    return round(p / tick) * tick if tick > 0 else p
