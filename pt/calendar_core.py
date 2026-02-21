#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import datetime as dt
from typing import Set


# Keep your holiday list here (same dates you already have)
US_MARKET_HOLIDAYS: Set[dt.date] = {
    dt.date(2025, 1, 1),
    dt.date(2025, 1, 20),
    dt.date(2025, 2, 17),
    dt.date(2025, 4, 18),
    dt.date(2025, 5, 26),
    dt.date(2025, 7, 4),
    dt.date(2025, 9, 1),
    dt.date(2025, 11, 27),
    dt.date(2025, 12, 25),
}


def weekend_lockout(now_ct: dt.datetime) -> bool:
    """
    True when we want to treat the market as 'weekend locked out'.

    - All day Saturday
    - Sunday before 17:00 CT (Globex open)
    - Sunday 17:00 CT and later -> trading allowed
    """
    wd = now_ct.weekday()  # Monday=0 ... Sunday=6
    t = now_ct.time()

    if wd == 5:  # Saturday
        return True
    if wd == 6 and t < dt.time(17, 0):  # Sunday before 17:00
        return True
    return False


def is_us_market_holiday(d: dt.date) -> bool:
    """Simple checker: date is in US_MARKET_HOLIDAYS."""
    return d in US_MARKET_HOLIDAYS
