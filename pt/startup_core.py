# startup_core.py
import os
import json
import sys
import time
import datetime as dt
from typing import Optional
from ib_insync import IB, Contract, StopOrder, LimitOrder
import utils
import order_core  # for round_to_tick if present

def maybe_daily_restart(now_ct: dt.datetime, logger, restart_json_path, cutoff_time):
    # your existing logic, but use params instead of global DAILY_RESTART_JSON/CT
    ...

def has_hedge_protection(ib: IB, con: Contract, net: int, logger) -> bool:
    # existing body
    ...

def attach_startup_protection(
    ib: IB,
    con: Contract,
    net: int,
    args,
    last_px: float,
    logger,
):
    # existing body
    ...

