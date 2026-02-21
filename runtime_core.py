# runtime_core.py
from typing import Tuple, Optional, List
import datetime as dt
import utils
import order_core
from position_core import compute_position

def runtime_safety_and_calendar(
    ib,
    con,
    now_ct: dt.datetime,
    last_px: float,
    day_risk,
    week_state,
    hb_path: str,
    logger,
    friday_flat_done: bool,
    friday_flat_date: Optional[dt.date],
    safety_halt_for_today: bool,
    last_ib_err,
    restart_ct,
) -> Tuple[
    int,           # net
    str,           # hb_pos_state
    bool,          # safety_halt_for_today
    Optional[dt.datetime],  # safety_last_ts
    bool,          # should_continue (True if we wrote a hb and want caller to continue loop)
    bool,          # friday_flat_done
    Optional[dt.date],      # friday_flat_date
    List[str],     # hard_caps
]:
    """
    Encapsulates:
    - naked-position guard
    - weekend flatten
    - Friday pre-close flatten
    - returns updated flags + whether caller should `continue`
    """
    ...

