from __future__ import annotations

# pt/session.py
#
# Session window helpers extracted from paper_trader.py.
# Side-effect free.

import datetime as dt
from typing import Dict, List, Optional, Tuple

try:
    # Preferred centralized time parsing
    from pt.utils_time import parse_hhmm  # type: ignore
except Exception:
    def parse_hhmm(s: str) -> dt.time:
        h, m = s.split(":")
        return dt.time(int(h), int(m))


def parse_ct_list(spec: str) -> List[dt.time]:
    spec = (spec or "").strip()
    if not spec:
        return [parse_hhmm("16:10")]
    out: List[dt.time] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            out.append(parse_hhmm(chunk))
        except Exception:
            pass
    out = sorted(list({t for t in out}))
    return out or [parse_hhmm("16:10")]


def within_session(now: dt.datetime, start_ct: str, end_ct: str) -> bool:
    t = now.time()
    a = parse_hhmm(start_ct)
    b = parse_hhmm(end_ct)
    if a <= b:
        return a <= t <= b
    return (t >= a) or (t <= b)


def parse_blackouts(spec: str) -> List[Tuple[dt.time, dt.time]]:
    out: List[Tuple[dt.time, dt.time]] = []
    spec = (spec or "").strip()
    if not spec:
        return out
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            a, b = chunk.split("-")
            out.append((parse_hhmm(a), parse_hhmm(b)))
        except Exception:
            pass
    return out


def in_tod_blackout(now: dt.datetime, blackouts: List[Tuple[dt.time, dt.time]]) -> bool:
    if not blackouts:
        return False
    t = now.time()
    for a, b in blackouts:
        if a <= b:
            if a <= t <= b:
                return True
        else:  # crosses midnight
            if (t >= a) or (t <= b):
                return True
    return False


def session_key_multi(now: dt.datetime, reset_times: List[dt.time]) -> str:
    t = now.time()
    idx_today = -1
    for i, ct_ in enumerate(reset_times):
        if t >= ct_:
            idx_today = i
        else:
            break
    if idx_today >= 0:
        base_date = now.date()
        seg = idx_today
    else:
        base_date = (now - dt.timedelta(days=1)).date()
        seg = len(reset_times) - 1
    return f"{base_date.strftime('%Y-%m-%d')}-S{seg}"


def reset_due_multi(now: dt.datetime, reset_times: List[dt.time], last_reset_marks: Dict[str, str]) -> Optional[str]:
    today = now.date().strftime("%Y-%m-%d")
    for ct_ in reset_times:
        label = ct_.strftime("%H:%M")
        if last_reset_marks.get(label) == today:
            continue
        if now.time() >= ct_:
            last_reset_marks[label] = today
            return f"{today}#{label}"
    return None
