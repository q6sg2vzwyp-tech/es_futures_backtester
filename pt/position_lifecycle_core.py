# position_lifecycle_core.py
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Optional


@dataclass
class PositionLifecycle:
    pos_entry_ct: Optional[dt.datetime] = None


def update_position_entry_time(life: PositionLifecycle, now_ct: dt.datetime, net: int) -> None:
    if net != 0:
        if life.pos_entry_ct is None:
            life.pos_entry_ct = now_ct
    else:
        life.pos_entry_ct = None


def maybe_flatten_on_age_cap(
    *,
    life: PositionLifecycle,
    now_ct: dt.datetime,
    net: int,
    pos_age_cap_sec: int,
    flatten_fn,
    logger,
) -> bool:
    """
    Returns True if it triggered a flatten attempt.
    """
    if net == 0:
        life.pos_entry_ct = None
        return False

    if life.pos_entry_ct is None:
        life.pos_entry_ct = now_ct
        return False

    if pos_age_cap_sec <= 0:
        return False

    age_sec = (now_ct - life.pos_entry_ct).total_seconds()
    if age_sec < float(pos_age_cap_sec):
        return False

    logger.info("[pos_age] flattening position after %.0fs (net=%s)", age_sec, net)
    try:
        flatten_fn()
    except Exception as e:
        logger.error("[pos_age] flatten_fn failed: %s", e)

    return True
