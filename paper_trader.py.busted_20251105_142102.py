#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ES Paper Trader (IBKR + ib_insync)
Helper utilities for TIF auditing/rebuilding.
"""
from __future__ import annotations

def audit_children_tif(ib: IB, pid: int, desired_tif: str, con: Contract) -> None:
    """
    Audit child OCO legs (stop/tp) under the given parent orderId `pid` and
    log any TIF mismatches vs `desired_tif`. We only log by default because
    some TWS builds enforce presets and block API TIF changes.
    """
    try:
        for tr in ib.trades():
            o = tr.order
            if getattr(o, "parentId", 0) != pid:
                continue
            have = (getattr(o, "tif", "") or "").upper()
            want = (desired_tif or "").upper()
            if have and want and have != want:
                log("tif_mismatch", orderId=o.orderId, parentId=pid, have=have, want=want)
    except Exception as e:
        log("tif_audit_error", pid=pid, err=str(e))

def rebuild_children_with_tif(
    ib: IB, con: Contract, pid: int, qty: int, side: str,
    stop: float, tp: float, desired_tif: str
) -> None:
    """
    Cancels existing children under parent `pid` and re-creates them with the desired TIF.
    The last child transmits=True to arm the OCO.
    """
    try:
        # Cancel existing children (stop & tp) for this parent
        for tr in ib.trades():
            o = tr.order
            if getattr(o, "parentId", 0) == pid and o.orderType in ("STP", "STP LMT", "LMT"):
                try:
                    ib.cancelOrder(o)
                except Exception as ce:
                    log("child_cancel_error", parentId=pid, orderId=o.orderId, err=str(ce))

        import time
        oca = f"OCA_{getattr(con,'conId',0)}_{int(time.time())}"
        action_child = "SELL" if side.upper() == "BUY" else "BUY"

        stp = StopOrder(action_child, qty, stop, tif=desired_tif, outsideRth=getattr(args, "outsideRth", False))
        stp.parentId = pid; stp.ocaGroup = oca; stp.ocaType = 1; stp.transmit = False
        tr_stp = ib.placeOrder(con, stp)

        lmt = LimitOrder(action_child, qty, tp, tif=desired_tif, outsideRth=getattr(args, "outsideRth", False))
        lmt.parentId = pid; lmt.ocaGroup = oca; lmt.ocaType = 1; lmt.transmit = True
        tr_tp = ib.placeOrder(con, lmt)

        log("oco_rebuilt_tif", parentId=pid, tif=desired_tif, stopId=tr_stp.order.orderId, tpId=tr_tp.order.orderId)
    except Exception as e:
        log("oco_rebuild_error", parentId=pid, err=str(e))
