#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pt_policy_flat.py

Policy flatten helper extracted from loop_core.py.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from ib_insync import MarketOrder

from trade_bridge import log_event, new_trade_id


def _attach_fill_event_logger(trade, *, event_name: str, trade_id: str, side: str, qty_hint: int, expected_px: Optional[float], logger=None):
    try:
        if trade is None:
            return
        filled_event = getattr(trade, "filledEvent", None)
        if filled_event is None:
            return

        def _on_filled(_tr):
            try:
                fill_px = None
                qty = qty_hint or 0

                try:
                    o = getattr(_tr, "orderStatus", None)
                    if o is not None:
                        ap = getattr(o, "avgFillPrice", None)
                        if ap is not None:
                            fill_px = float(ap)
                        fq = getattr(o, "filled", None)
                        if fq is not None:
                            qty = int(float(fq))
                except Exception:
                    pass

                try:
                    fills = getattr(_tr, "fills", None)
                    if fills:
                        last = fills[-1]
                        ex = getattr(last, "execution", None)
                        if ex is not None and getattr(ex, "price", None) is not None:
                            fill_px = float(ex.price)
                        if ex is not None and getattr(ex, "shares", None) is not None:
                            qty = int(float(ex.shares))
                except Exception:
                    pass

                log_event(
                    event_name,
                    trade_id,
                    side=side,
                    qty=int(qty or 0),
                    fill_px=fill_px,
                    expected_px=expected_px,
                    reason="pt_policy_flat_fill",
                    extra={"source": "pt_policy_flat"},
                )
            except Exception:
                pass

        filled_event += _on_filled
    except Exception:
        pass


def place_policy_flat_market(*, ib, con, current_net: int, reason: str, logger) -> None:
    if current_net == 0:
        return
    action = "SELL" if current_net > 0 else "BUY"
    qty = int(round(abs(current_net)))

    try:
        logger.warning("[policy_flat] sending %s %s @ MKT (net=%s)", action, qty, current_net)
    except Exception:
        pass

    order = MarketOrder(action, qty)
    tid = new_trade_id("PFLAT")
    try:
        order.orderRef = tid
    except Exception:
        pass

    try:
        log_event(
            "policy_flat_submit",
            tid,
            side=action,
            qty=int(qty or 0),
            expected_px=None,
            reason=str(reason or "loop_core_policy_flat"),
            net=int(current_net),
        )
    except Exception:
        pass

    tr = ib.placeOrder(con, order)
    try:
        _attach_fill_event_logger(tr, event_name="policy_flat_fill", trade_id=tid, side=action, qty_hint=qty, expected_px=None, logger=logger)
    except Exception:
        pass


def apply_day_policy_block(*, ctx: Dict[str, Any], now_ct, net: int) -> Tuple[Any, List[str]]:
    args = ctx["args"]
    logger = ctx["logger"]
    from day_policy_core import apply_day_policies
    from order_core import flatten_until_flat

    AUTO_FLAT_CT = ctx["AUTO_FLAT_CT"]
    PRE_CLOSE_SWEEP_CT = ctx.get("PRE_CLOSE_SWEEP_CT", None)
    WEEKEND_FLATTEN = bool(ctx.get("WEEKEND_FLATTEN", False))
    is_us_market_holiday = ctx["is_us_market_holiday"]
    day_policy_state = ctx["day_policy_state"]

    def _flatten_all() -> None:
        ok = flatten_until_flat(ctx["ib"], ctx["con"], logger=logger, max_attempts=10, sleep_sec=1.0)
        if not ok:
            raise RuntimeError("flatten_until_flat returned False")

    policy_res = apply_day_policies(
        now_ct=now_ct,
        net=net,
        auto_flat_ct=AUTO_FLAT_CT,
        preclose_sweep_ct=PRE_CLOSE_SWEEP_CT,
        weekend_flatten=WEEKEND_FLATTEN,
        place_orders=bool(getattr(args, "place_orders", False)),
        is_us_market_holiday=is_us_market_holiday,
        flatten_all=_flatten_all,
        place_market_flat=lambda current_net: place_policy_flat_market(
            ib=ctx["ib"], con=ctx["con"], current_net=int(current_net), reason="loop_core_policy_flat", logger=logger
        ),
        logger=logger,
        state=day_policy_state,
    )

    hard_caps: List[str] = []
    if bool(ctx.get("safety_halt_for_today", False)):
        hard_caps.append("safety_halt_for_today")
    if policy_res is not None:
        try:
            hard_caps.extend(list(getattr(policy_res, "hard_caps", []) or []))
        except Exception:
            pass

    return policy_res, hard_caps
