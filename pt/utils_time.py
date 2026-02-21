from __future__ import annotations

# pt/utils_time.py
#
# Centralized time helpers for ES paper trader.

import datetime as dt


def utc_now_str() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def ct_now() -> dt.datetime:
    # assumes machine local time is Central Time
    return dt.datetime.now()


def parse_hhmm(s: str) -> dt.time:
    h, m = s.split(":")
    return dt.time(int(h), int(m))
