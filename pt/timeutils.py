from __future__ import annotations

import datetime as dt

# Cached tzinfo holder. We keep it module-global to avoid repeated ZoneInfo construction.
_CT_TZ = None


def parse_hhmm(s: str) -> dt.time:
    """Parse 'HH:MM' into a datetime.time."""
    try:
        s = (s or "").strip()
        hh, mm = s.split(":", 1)
        return dt.time(int(hh), int(mm))
    except Exception:
        return dt.time(0, 0)


def ct_now() -> dt.datetime:
    """
    Current time in America/Chicago, returned as naive local datetime (for legacy code paths).

    We intentionally drop tzinfo to preserve existing comparisons with naive datetimes elsewhere
    in the codebase.
    """
    global _CT_TZ
    try:
        if _CT_TZ is None:
            from zoneinfo import ZoneInfo  # py3.9+
            _CT_TZ = ZoneInfo("America/Chicago")

        # timezone-aware -> convert -> drop tzinfo (naive local)
        return dt.datetime.now(_CT_TZ).replace(tzinfo=None)
    except Exception:
        # last-resort fallback: local system time (naive)
        return dt.datetime.now()
