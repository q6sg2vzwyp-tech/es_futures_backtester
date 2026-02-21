#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pt_ib_handlers.py

IB event handlers + lightweight shared state for paper_trader.

Provides:
- IBEventState: holds last_fill_price / last_fill_ts / last_ib_err
- attach_ib_handlers(ib, logger, state): wires ib.errorEvent and ib.execDetailsEvent

Design goals:
- Keep logic equivalent to current inline handlers
- Avoid leaking closures and scattered variables
- Keep decoding/formatting consistent
"""

from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class IBEventState:
    last_fill_price: Optional[float] = None
    last_fill_ts: Optional[float] = None
    last_ib_err: Optional[Dict[str, Any]] = None

    def clear_stale_error(self, *, decay_sec: float = 120.0) -> None:
        """
        Clears last_ib_err if it is older than decay_sec.
        Safe to call in-loop.
        """
        if not self.last_ib_err:
            return
        try:
            err_ts = dt.datetime.fromisoformat(self.last_ib_err.get("ts", "1970-01-01T00:00:00"))
        except Exception:
            err_ts = dt.datetime.now()
        if (dt.datetime.now() - err_ts).total_seconds() > float(decay_sec):
            self.last_ib_err = None

    def recently_filled(self, *, grace_sec: float = 1.0) -> bool:
        """
        True if last fill happened within grace_sec.
        """
        if self.last_fill_ts is None:
            return False
        return (time.time() - float(self.last_fill_ts)) < float(grace_sec)


def attach_ib_handlers(*, ib, logger, state: IBEventState) -> None:
    """
    Attaches handlers to:
      - ib.errorEvent
      - ib.execDetailsEvent

    Note: this function intentionally does not return handler refs.
    IB insync keeps references via event lists.
    """

    def on_ib_error(reqId, errorCode, errorString, contract):
        try:
            code = int(errorCode)
        except Exception:
            code = -1
        msg = str(errorString)

        state.last_ib_err = {
            "ts": dt.datetime.now().isoformat(timespec="seconds"),
            "code": code,
            "msg": msg,
            "reqId": int(reqId) if reqId is not None else -1,
        }

        # Status / benign informational codes (keep your current semantics)
        if code in (2103, 2104, 2105, 2106, 2119, 2157, 2158):
            logger.info("[IB_STATUS] reqId=%s code=%s msg=%s", reqId, code, msg)
            return
        if code in (1100, 1102):
            logger.error("[IB_CONN] reqId=%s code=%s msg=%s", reqId, code, msg)
            return
        if code == 2109:
            logger.info("[IB_ORDER_WARN] reqId=%s code=%s msg=%s", reqId, code, msg)
            return
        if code == 10148:
            if "state: Cancelled" in msg or "Completed" in msg:
                logger.debug("[IB_WARN_SUPPRESSED] reqId=%s code=%s msg=%s", reqId, code, msg)
            else:
                logger.warning("[IB_WARN] reqId=%s code=%s msg=%s", reqId, code, msg)
            return
        if code == 10147:
            logger.warning("[IB_WARN] reqId=%s code=%s msg=%s", reqId, code, msg)
            return
        if code == 202:
            logger.info("[IB_CANCEL] reqId=%s code=%s msg=%s", reqId, code, msg)
            return

        logger.error("[IB_ERROR] reqId=%s code=%s msg=%s", reqId, code, msg)

    def on_fill(trade, fill):
        try:
            exec_obj = getattr(fill, "execution", None)
            if exec_obj is None:
                raise AttributeError("fill.execution is None")

            px = float(getattr(exec_obj, "price", 0.0))
            qty = float(getattr(exec_obj, "shares", 0.0))

            if px > 0:
                state.last_fill_price = px
                state.last_fill_ts = time.time()
                logger.info("[fill] price=%s qty=%s", px, qty)
            else:
                logger.warning("[fill] got non-positive price=%s qty=%s", px, qty)
        except Exception as e:
            logger.error("[fill_handler] failed: %s", e)

    # Wire up events
    ib.errorEvent += on_ib_error
    ib.execDetailsEvent += on_fill
