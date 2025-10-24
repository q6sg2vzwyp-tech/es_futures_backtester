from datetime import date, datetime, time
from zoneinfo import ZoneInfo


def is_holiday(d: date, holidays: list[str]) -> bool:
    return d.strftime("%Y-%m-%d") in {h.strip() for h in holidays or []}


def is_session_open(now: datetime, tz: str, open_hm: str, close_hm: str) -> bool:
    z = ZoneInfo(tz)
    nl = now.astimezone(z)
    oh = time.fromisoformat(open_hm)
    ch = time.fromisoformat(close_hm)
    # Treat the market as "closed" only in the gap between close..open (Globex overlap).
    return not (nl.time() >= ch and nl.time() < oh)
