# ib_events.py
from __future__ import annotations

import time
import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class IBEventState:
    last_fill_price: Optional[float] = None
    last_fill_ts: Optional[float] = None
    last_ib_err: Optional[Dict[str, Any]] = None


def attach_ib_handlers(ib, logger) -> IBEventState:
    state = IBEventState()

    def on_ib_error(reqId, errorCode, errorString, contract):
        code = int(errorCode)
        msg = str(errorString)

        state.last_ib_err = {
            "ts": dt.datetime.now().isoformat(timespec="seconds"),
            "code": code,
            "msg": msg,
            "reqId": int(reqId),
        }

        # status / connectivity noise
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

    ib.errorEvent += on_ib_error
    ib.execDetailsEvent += on_fill
    return state


def decay_last_ib_error(state: IBEventState, decay_sec: float) -> None:
    if state.last_ib_err is None:
        return

    try:
        err_ts = dt.datetime.fromisoformat(state.last_ib_err.get("ts", "1970-01-01T00:00:00"))
    except Exception:
        err_ts = dt.datetime.now()

    if (dt.datetime.now() - err_ts).total_seconds() > float(decay_sec):
        state.last_ib_err = None
