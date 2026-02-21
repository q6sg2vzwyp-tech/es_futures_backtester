# md_core.py
from __future__ import annotations

import time
from typing import Optional


def get_last_px(ticker) -> Optional[float]:
    last_price = getattr(ticker, "last", None) or ticker.marketPrice()
    if last_price is None:
        return None
    try:
        px = float(last_price)
    except Exception:
        return None
    if px <= 0:
        return None
    return px


def get_last_px_or_wait(ticker, logger, sleep_sec: float = 1.0) -> float:
    """
    Blocks until a valid last price is available.
    """
    while True:
        px = get_last_px(ticker)
        if px is not None:
            return px
        logger.warning("[md] no last price yet (ticker.last/marketPrice None); waiting...")
        time.sleep(float(sleep_sec))
