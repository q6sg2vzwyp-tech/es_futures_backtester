#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import math
import logging
import traceback
import datetime as dt
from typing import Any, Dict, List, Optional, Tuple

# Chicago time (America/Chicago, UTC-6 / -5 with DST, but we keep it simple)
CT_OFFSET = -6
CT_TZ = dt.timezone(dt.timedelta(hours=CT_OFFSET))


def ct_now() -> dt.datetime:
    """Current time in Central Time."""
    return dt.datetime.now(tz=CT_TZ)


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def ensure_dir(path: str) -> None:
    if not path:
        return
    os.makedirs(path, exist_ok=True)


def setup_logger(log_dir: str, name: str = "es_paper") -> logging.Logger:
    ensure_dir(log_dir)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fh = logging.FileHandler(os.path.join(log_dir, f"{dt.datetime.now():%Y%m%d}.log"), encoding="utf-8")
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)

        fmt = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


def write_json_line(path: str, payload: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def save_json(path: str, payload: Any) -> None:
    ensure_dir(os.path.dirname(path))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def load_json(path: str, default: Any = None) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def parse_ct_time(spec: str) -> dt.time:
    """Parse 'HH:MM' in CT."""
    parts = spec.strip().split(":")
    h = int(parts[0])
    m = int(parts[1]) if len(parts) > 1 else 0
    return dt.time(hour=h, minute=m)


def parse_ct_list(spec: str) -> List[dt.time]:
    out: List[dt.time] = []
    for item in (spec or "").split(","):
        item = item.strip()
        if not item:
            continue
        out.append(parse_ct_time(item))
    return out


def iso_week_id(d: dt.date) -> str:
    y, w, _ = d.isocalendar()
    return f"{y}-W{int(w):02d}"


def is_weekend(now: Optional[dt.datetime] = None) -> bool:
    now = now or ct_now()
    return now.weekday() >= 5  # 5=Sat, 6=Sun


def is_friday(now: Optional[dt.datetime] = None) -> bool:
    now = now or ct_now()
    return now.weekday() == 4  # 4 = Friday


import datetime as dt

def in_time_window(now: dt.datetime, start: dt.time, end: dt.time) -> bool:
    """
    Return True if `now` (CT) is between start/end (inclusive).

    Works whether `now` is offset-aware (has tzinfo) or naive,
    and handles windows that can wrap across midnight.
    """
    # Convert `now` to a time-of-day, drop tzinfo
    if isinstance(now, dt.datetime):
        t = now.time()
    elif isinstance(now, dt.time):
        t = now
    else:
        raise TypeError("now must be datetime or time")

    # Make everything naive so comparisons are legal
    t = t.replace(tzinfo=None)
    start = start.replace(tzinfo=None)
    end = end.replace(tzinfo=None)

    # Normal window (e.g. 08:30â€“15:00)
    if start <= end:
        return start <= t <= end
    # Overnight window (e.g. 17:00â€“07:00)
    else:
        return (t >= start) or (t <= end)



import json
import os

import json
import os

import json
import os
import time

def write_heartbeat(path: str, payload: dict) -> None:
    """
    Atomically write a JSON heartbeat file.

    On Windows, another process (viewer/monitor) may hold the target file
    open with an exclusive lock, causing os.replace() to raise PermissionError.
    We treat that as a non-fatal, best-effort failure: log if possible and
    skip the replace for this loop.
    """
    # Make sure directory exists
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

    tmp = path + ".tmp"

    # Write JSON to temp file first
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, sort_keys=True)

    try:
        # Atomic replace on POSIX; best-effort on Windows
        os.replace(tmp, path)
    except PermissionError:
        # Another process is probably reading/locking heartbeat.txt.
        # Best-effort: remove the temp file and move on; don't crash the bot.
        try:
            os.remove(tmp)
        except OSError:
            pass
        # Optional: print a debug message; avoid importing logger here.
        # print(f"[write_heartbeat] PermissionError replacing {path}, skipping this update.")
    except OSError:
        # Any other unexpected filesystem issue: best-effort clean-up
        try:
            os.remove(tmp)
        except OSError:
            pass




def exception_to_str(e: BaseException) -> str:
    return "".join(traceback.format_exception(type(e), e, e.__traceback__))

