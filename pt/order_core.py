#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
order_core.py  (v3.5 - bracket cleanup + consistent event attribution)

Core order helpers for ES Paper Trader

Goals:
- Parent entry is MARKET to avoid repricing / timing issues.
- Children (STOP + TARGET) live in a single OCA group per contract.
- We NEVER intentionally leave a live position unprotected.
- We keep at most one pair of child orders (1 stop, 1 target) per net position.
- Backwards-compatible `reconcile_orphans` shim so older paper_trader code works.

NEW in v3.4:
- maybe_enter_trade(...): moves the "REAL ENTRY PATH" out of paper_trader.py
  so paper_trader can shrink without losing behavior.

NEW in v3.5:
- Fixes duplicate parent placeOrder bug
- Removes undefined-variable blocks (arm/side/trade_id) that could crash silently
- Optional but recommended: consistent arm + extra attribution across submits/fills
"""

import time
import uuid
from typing import List, Optional, Tuple

from ib_insync import IB, Contract, Trade, Order, MarketOrder, StopOrder, LimitOrder

# For entry sizing (keeps paper_trader thinner)
from position_core import dynamic_contracts

# Execution/event logging (append-only, non-blocking)
from trade_bridge import log_event, new_trade_id


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _log(logger, level: str, msg: str) -> None:
    if logger is None:
        return
    fn = getattr(logger, level, None)
    if fn is None:
        return
    try:
        fn(msg)
    except Exception:
        pass


def _attach_fill_event_logger(
    trade,
    *,
    event_name: str,
    trade_id: str,
    arm: str = "",
    side: str = "?",
    qty_hint: int = 0,
    expected_px: float | None = None,
    logger=None,
):
    """Attach a filledEvent handler that logs fill facts into trade_events.csv."""
    try:
        if trade is None:
            return
        filled_event = getattr(trade, "filledEvent", None)
        if filled_event is None:
            return

        def _handler(*args, **kwargs):
            try:
                t = trade
                os_ = getattr(t, "orderStatus", None)

                avg_px = None
                filled_qty = None
                try:
                    avg_px = float(getattr(os_, "avgFillPrice", None)) if os_ is not None else None
                except Exception:
                    avg_px = None
                try:
                    filled_qty = int(getattr(os_, "filled", None)) if os_ is not None else None
                except Exception:
                    filled_qty = None

                order_id = None
                try:
                    order_id = getattr(getattr(t, "order", None), "orderId", None)
                except Exception:
                    order_id = None

                log_event(
                    event_name,
                    trade_id,
                    order_id=str(order_id) if order_id is not None else "",
                    arm=arm or "",
                    side=side or "?",
                    qty=int(filled_qty if filled_qty is not None else (qty_hint or 0)),
                    fill_px=avg_px,
                    expected_px=expected_px,
                    commission=None,
                    reason="fill",
                    # Optional but recommended: consistent attribution for fills too
                    extra={"source": "order_core", "mode": event_name},
                )
            except Exception as e:
                if logger is not None:
                    try:
                        logger.error(f"[event_log] fill handler failed: {e}")
                    except Exception:
                        pass

        try:
            filled_event += _handler
        except Exception:
            pass

    except Exception:
        return


def _record_bracket(logger, tag: str, **fields) -> None:
    """
    Structured 'recorder' for anything related to brackets / protective orders.
    """
    if logger is None:
        return
    try:
        payload = ", ".join(f"{k}={v}" for k, v in fields.items())
        logger.info(f"[bracket_{tag}] {payload}")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Order classification / cancellation helpers
# ---------------------------------------------------------------------------

def _classify_order(o: Order) -> Optional[str]:
    """Classify an order as 'stop', 'target', or None."""
    if o is None:
        return None

    t = (o.orderType or "").upper()

    if t in ("STP", "STOP", "STP LMT", "STP_LIMIT", "STOP_LIMIT"):
        return "stop"

    if t in ("LMT", "LIMIT"):
        return "target"

    return None


def _cancel_orders(ib: IB, orders: List[Order], logger=None) -> None:
    """Safely cancel a list of orders."""
    for o in orders:
        try:
            oid = getattr(o, "orderId", None)
            _log(
                logger,
                "info",
                f"[order_core] cancelling orderId={oid}, type={o.orderType}, qty={o.totalQuantity}",
            )
            ib.cancelOrder(o)
        except Exception as e:
            _log(
                logger,
                "error",
                f"[order_core] error cancelling orderId={getattr(o, 'orderId', None)}: {e}",
            )


def _make_oca_group(contract: Contract) -> str:
    """Build a unique-ish OCA group name per contract."""
    con_id = getattr(contract, "conId", 0)
    return f"ES_OCA_{con_id}_{int(time.time())}_{uuid.uuid4().hex[:6]}"


# ---------------------------------------------------------------------------
# Bracket / protected entry
# ---------------------------------------------------------------------------

def place_protected_entry(
    ib: IB,
    contract: Contract,
    action: str,
    qty: int,
    stop_px: float,
    target_px: float,
    tif: str = "GTC",
    outsideRth: bool = True,
    logger=None,
    px_hint: Optional[float] = None,
    arm: Optional[str] = None,  # OPTIONAL BUT RECOMMENDED: attribution
) -> Tuple[bool, Optional[int], Optional[int], Optional[int]]:
    """
    Place a MARKET entry with attached STOP and TARGET in a single OCA group.
    Returns: (ok, parent_order_id, stop_order_id, target_order_id)
    """
    side = action.upper()
    if side not in ("BUY", "SELL"):
        raise ValueError(f"action must be BUY or SELL, got: {action}")

    child_action = "SELL" if side == "BUY" else "BUY"
    arm_s = str(arm or "")

    if px_hint is not None:
        _log(logger, "info", f"[place_protected_entry] px_hint={px_hint}")

    parent = MarketOrder(
        action=side,
        totalQuantity=qty,
        tif=tif,
        outsideRth=outsideRth,
    )

    oca = _make_oca_group(contract)

    stop = StopOrder(
        action=child_action,
        totalQuantity=qty,
        stopPrice=stop_px,
        tif=tif,
        outsideRth=outsideRth,
    )
    stop.ocaGroup = oca
    stop.ocaType = 1

    target = LimitOrder(
        action=child_action,
        totalQuantity=qty,
        lmtPrice=target_px,
        tif=tif,
        outsideRth=outsideRth,
    )
    target.ocaGroup = oca
    target.ocaType = 1

    _record_bracket(
        logger,
        "entry_build",
        side=side,
        qty=qty,
        stop_px=stop_px,
        target_px=target_px,
        tif=tif,
        outsideRth=outsideRth,
        oca=oca,
        px_hint=px_hint,
        arm=arm_s,
    )

    _log(
        logger,
        "info",
        f"[place_protected_entry] action={side}, qty={qty}, stop={stop_px}, target={target_px}, oca={oca}",
    )

    parent_id = stp_id = tgt_id = None
    trade_id = None

    try:
        trade_id = new_trade_id("BRK")
        try:
            parent.orderRef = trade_id
            stop.orderRef = trade_id
            target.orderRef = trade_id
        except Exception:
            pass

        # Entry submit (single, authoritative)
        try:
            log_event(
                "entry_submit",
                trade_id,
                arm=arm_s,
                side=side,
                qty=int(qty or 0),
                expected_px=px_hint,
                reason="protected_entry",
                stop_px=stop_px,
                target_px=target_px,
                oca=oca,
                extra={"source": "order_core", "mode": "protected_entry"},
            )
        except Exception:
            pass

        # Place parent ONCE (fixes duplicate parent bug)
        parent_trade: Trade = ib.placeOrder(contract, parent)
        parent_id = getattr(parent_trade.order, "orderId", None)

        try:
            _attach_fill_event_logger(
                parent_trade,
                event_name="entry_fill",
                trade_id=trade_id,
                arm=arm_s,
                side=side,
                qty_hint=qty,
                expected_px=px_hint,
                logger=logger,
            )
        except Exception:
            pass

        # Children submits + fill loggers
        stop_trade: Trade = ib.placeOrder(contract, stop)
        stp_id = getattr(stop_trade.order, "orderId", None)
        try:
            log_event(
                "stop_submit",
                trade_id,
                arm=arm_s,
                side=child_action,
                qty=int(qty or 0),
                expected_px=stop_px,
                reason="bracket_stop",
                oca=oca,
                extra={"source": "order_core", "mode": "protected_entry"},
            )
            _attach_fill_event_logger(
                stop_trade,
                event_name="stop_fill",
                trade_id=trade_id,
                arm=arm_s,
                side=child_action,
                qty_hint=qty,
                expected_px=stop_px,
                logger=logger,
            )
        except Exception:
            pass

        target_trade: Trade = ib.placeOrder(contract, target)
        tgt_id = getattr(target_trade.order, "orderId", None)
        try:
            log_event(
                "target_submit",
                trade_id,
                arm=arm_s,
                side=child_action,
                qty=int(qty or 0),
                expected_px=target_px,
                reason="bracket_target",
                oca=oca,
                extra={"source": "order_core", "mode": "protected_entry"},
            )
            _attach_fill_event_logger(
                target_trade,
                event_name="target_fill",
                trade_id=trade_id,
                arm=arm_s,
                side=child_action,
                qty_hint=qty,
                expected_px=target_px,
                logger=logger,
            )
        except Exception:
            pass

        _log(logger, "info", f"[place_protected_entry] placed parent_id={parent_id}, stop_id={stp_id}, target_id={tgt_id}")

        _record_bracket(
            logger,
            "entry_placed",
            parent_id=parent_id,
            stop_id=stp_id,
            target_id=tgt_id,
            side=side,
            qty=qty,
            stop_px=stop_px,
            target_px=target_px,
            oca=oca,
            arm=arm_s,
            trade_id=trade_id,
        )

        return True, parent_id, stp_id, tgt_id

    except Exception as e:
        _log(logger, "error", f"[place_protected_entry] EXCEPTION: {e}")
        _record_bracket(
            logger,
            "entry_error",
            side=side,
            qty=qty,
            stop_px=stop_px,
            target_px=target_px,
            oca=oca,
            arm=arm_s,
            trade_id=trade_id,
            error=str(e),
        )
        return False, parent_id, stp_id, tgt_id


# Compatibility alias if old code calls this name
enter_market_with_children = place_protected_entry


# ---------------------------------------------------------------------------
# Protective order reconciliation
# ---------------------------------------------------------------------------

def reconcile_protective_orders(
    ib: IB,
    contract: Contract,
    net_qty: int,
    stop_px: Optional[float],
    target_px: Optional[float],
    logger=None,
) -> None:
    """
    Ensure that protective orders (STOP + TARGET) match the current position.
    """
    try:
        con_id = getattr(contract, "conId", None)
        if con_id is None:
            _log(logger, "warning", "[reconcile_protective_orders] contract has no conId, aborting.")
            return

        stop_orders: List[Order] = []
        target_orders: List[Order] = []

        for tr in ib.openTrades():
            try:
                if getattr(tr.contract, "conId", None) != con_id:
                    continue
                o = tr.order
                kind = _classify_order(o)
                if kind == "stop":
                    stop_orders.append(o)
                elif kind == "target":
                    target_orders.append(o)
            except Exception as inner_e:
                _log(logger, "error", f"[reconcile_protective_orders] error scanning trade: {inner_e}")

        if net_qty == 0 and not stop_orders and not target_orders:
            _log(logger, "debug", "[reconcile_protective_orders] flat & clean: net_qty=0, stops=0, targets=0")
        else:
            _log(
                logger,
                "info",
                f"[reconcile_protective_orders] net_qty={net_qty}, stops={len(stop_orders)}, targets={len(target_orders)}, desired stop_px={stop_px}, target_px={target_px}",
            )

        if net_qty == 0:
            to_cancel = stop_orders + target_orders
            if to_cancel:
                _log(logger, "info", f"[reconcile_protective_orders] flat position, cancelling {len(to_cancel)} protective orders.")
                for o in to_cancel:
                    _record_bracket(
                        logger,
                        "trim_flat",
                        order_id=getattr(o, "orderId", None),
                        order_type=getattr(o, "orderType", None),
                        qty=getattr(o, "totalQuantity", None),
                    )
                _cancel_orders(ib, to_cancel, logger=logger)
            return

        def keep_one_closest(orders: List[Order], desired_px: Optional[float], label: str) -> None:
            if not orders:
                return
            if desired_px is None or len(orders) == 1:
                return

            def get_price(o: Order) -> float:
                t = (o.orderType or "").upper()
                if t in ("STP", "STOP", "STP LMT", "STP_LIMIT", "STOP_LIMIT") or t.startswith("STP"):
                    return float(o.auxPrice)
                return float(o.lmtPrice)

            sorted_orders = sorted(orders, key=lambda o: abs(get_price(o) - desired_px))
            keep = sorted_orders[0]
            kill = sorted_orders[1:]

            if kill:
                _log(logger, "info", f"[reconcile_protective_orders] trimming {label}: keeping orderId={keep.orderId}, cancelling {len(kill)} others.")
                for o in kill:
                    _record_bracket(
                        logger,
                        f"trim_{label}",
                        keep_id=getattr(keep, "orderId", None),
                        cancel_id=getattr(o, "orderId", None),
                        desired_px=desired_px,
                        actual_px=get_price(o),
                    )
                _cancel_orders(ib, kill, logger=logger)

        keep_one_closest(stop_orders, stop_px, label="stops")
        keep_one_closest(target_orders, target_px, label="targets")

    except Exception as e:
        _log(logger, "error", f"[reconcile_protective_orders] EXCEPTION: {e}")


# ---------------------------------------------------------------------------
# Flatten helpers
# ---------------------------------------------------------------------------

def flatten_all(ib: IB, contract: Contract, logger=None) -> None:
    """Market-flatten all ES position for `contract` (one-shot)."""
    try:
        con_id = getattr(contract, "conId", None)
        if con_id is None:
            _log(logger, "warning", "[flatten_all] contract has no conId; aborting.")
            return

        try:
            for tr in ib.openTrades():
                o = getattr(tr, "order", None)
                if o is None:
                    continue
                if getattr(getattr(tr, "contract", None), "conId", None) != con_id:
                    continue
                ib.cancelOrder(o)
        except Exception:
            pass

        net = 0
        for pos in ib.positions():
            try:
                if getattr(pos.contract, "conId", None) == con_id:
                    net += int(pos.position)
            except Exception:
                continue

        if net == 0:
            _log(logger, "info", "[flatten_all] net position already flat; nothing to do.")
            return

        action = "SELL" if net > 0 else "BUY"
        qty = abs(net)

        _log(logger, "warning", f"[flatten_all] sending {action} {qty} @ MKT to flatten net={net}")

        mkt = MarketOrder(action=action, totalQuantity=qty)
        mkt.tif = "DAY"
        mkt.outsideRth = True

        tid = new_trade_id("FLAT")
        try:
            mkt.orderRef = tid
        except Exception:
            pass

        # OPTIONAL BUT RECOMMENDED: include attribution on flatten submit
        try:
            log_event(
                "policy_flat_submit",
                tid,
                arm="",
                side=action,
                qty=int(qty or 0),
                expected_px=None,
                reason="flatten_all",
                extra={"source": "order_core", "mode": "flatten_all"},
            )
        except Exception:
            pass

        tr = ib.placeOrder(contract, mkt)
        try:
            _attach_fill_event_logger(
                tr,
                event_name="policy_flat_fill",
                trade_id=tid,
                arm="",
                side=action,
                qty_hint=qty,
                expected_px=None,
                logger=logger,
            )
        except Exception:
            pass

    except Exception as e:
        _log(logger, "error", f"[flatten_all] EXCEPTION: {e}")


def flatten_until_flat(
    ib: IB,
    contract: Contract,
    *,
    logger=None,
    max_attempts: int = 8,
    sleep_sec: float = 1.0,
) -> bool:
    """Retry flatten until position is flat or attempts exhausted."""
    try:
        con_id = getattr(contract, "conId", None)
        if con_id is None:
            _log(logger, "warning", "[flatten_until_flat] contract has no conId; aborting.")
            return False

        for attempt in range(1, int(max_attempts) + 1):
            net = 0
            for pos in ib.positions():
                try:
                    if getattr(pos.contract, "conId", None) == con_id:
                        net += int(pos.position)
                except Exception:
                    continue

            if net == 0:
                _log(logger, "warning", f"[flatten_until_flat] flat after attempt {attempt}/{max_attempts}")
                return True

            _log(logger, "warning", f"[flatten_until_flat] attempt {attempt}/{max_attempts} net={net}")
            flatten_all(ib, contract, logger=logger)

            try:
                ib.sleep(float(sleep_sec))
            except Exception:
                time.sleep(float(sleep_sec))

        net_final = 0
        for pos in ib.positions():
            try:
                if getattr(pos.contract, "conId", None) == con_id:
                    net_final += int(pos.position)
            except Exception:
                continue

        _log(logger, "error", f"[flatten_until_flat] FAILED net={net_final} after {max_attempts} attempts")
        return (net_final == 0)

    except Exception as e:
        _log(logger, "error", f"[flatten_until_flat] EXCEPTION: {e}")
        return False


# ---------------------------------------------------------------------------
# Orphan reconciler (shim used by paper_trader main loop)
# ---------------------------------------------------------------------------

def reconcile_orphans(
    ib: IB,
    contract: Contract,
    net_qty: int,
    logger=None,
) -> int:
    """Backwards-compat shim for older paper_trader code."""
    try:
        con_id = getattr(contract, "conId", None)
        if con_id is None:
            _log(logger, "warning", "[reconcile_orphans] contract has no conId; aborting.")
            return 0

        if net_qty != 0:
            _log(logger, "debug", f"[reconcile_orphans] net_qty={net_qty}, skipping orphan sweep (only runs when flat).")
            return 0

        stop_orders: List[Order] = []
        target_orders: List[Order] = []

        for tr in ib.openTrades():
            try:
                if getattr(tr.contract, "conId", None) != con_id:
                    continue
                o = tr.order
                kind = _classify_order(o)
                if kind == "stop":
                    stop_orders.append(o)
                elif kind == "target":
                    target_orders.append(o)
            except Exception as inner_e:
                _log(logger, "error", f"[reconcile_orphans] error scanning trade: {inner_e}")

        all_orphans = stop_orders + target_orders

        if not all_orphans:
            _log(logger, "debug", "[reconcile_orphans] flat & no protective orders; nothing to cancel.")
            return 0

        _log(logger, "info", f"[reconcile_orphans] flat net=0, cancelling {len(all_orphans)} orphan protective orders.")

        for o in all_orphans:
            _record_bracket(
                logger,
                "orphan_cancel",
                order_id=getattr(o, "orderId", None),
                order_type=getattr(o, "orderType", None),
                qty=getattr(o, "totalQuantity", None),
            )

        _cancel_orders(ib, all_orphans, logger=logger)
        return len(all_orphans)

    except Exception as e:
        _log(logger, "error", f"[reconcile_orphans] EXCEPTION: {e}")
        return 0


# ---------------------------------------------------------------------------
# Guard naked position (fix undefined vars + add attribution)
# ---------------------------------------------------------------------------

def guard_naked_position(
    ib: IB,
    contract: Contract,
    net_qty: int,
    last_px: float,
    args,
    logger=None,
) -> None:
    """
    Attach a protective STOP + TARGET to an existing net position.
    """
    try:
        if net_qty == 0:
            return
        if last_px is None:
            _log(logger, "warning", "[guard_naked_position] last_px is None; skipping.")
            return

        qty = abs(int(net_qty))
        if qty <= 0:
            return

        child_action = "SELL" if net_qty > 0 else "BUY"

        risk_ticks = float(getattr(args, "risk_ticks", 12) or 12)
        tick_size = float(getattr(args, "tick_size", 0.25) or 0.25)
        tp_R = float(getattr(args, "tp_R", 1.0) or 1.0)

        stop_dist = risk_ticks * tick_size
        tp_dist = stop_dist * tp_R

        if stop_dist <= 0 or tp_dist <= 0:
            _log(logger, "warning", f"[guard_naked_position] invalid stop/tp distances: stop_dist={stop_dist}, tp_dist={tp_dist}; skipping.")
            return

        if net_qty > 0:
            stop_px = last_px - stop_dist
            target_px = last_px + tp_dist
        else:
            stop_px = last_px + stop_dist
            target_px = last_px - tp_dist

        oca = _make_oca_group(contract)

        _record_bracket(
            logger,
            "guard_build",
            net_qty=net_qty,
            qty=qty,
            last_px=last_px,
            stop_px=stop_px,
            target_px=target_px,
            oca=oca,
        )

        stop = StopOrder(action=child_action, totalQuantity=qty, stopPrice=stop_px, tif="GTC", outsideRth=True)
        stop.ocaGroup = oca
        stop.ocaType = 1

        target = LimitOrder(action=child_action, totalQuantity=qty, lmtPrice=target_px, tif="GTC", outsideRth=True)
        target.ocaGroup = oca
        target.ocaType = 1

        _log(logger, "info", f"[guard_naked_position] net={net_qty}, qty={qty}, stop={stop_px}, target={target_px}, oca={oca}")

        trade_id = new_trade_id("GUARD")
        try:
            stop.orderRef = trade_id
            target.orderRef = trade_id
        except Exception:
            pass

        stop_trade: Trade = ib.placeOrder(contract, stop)
        stop_id = getattr(stop_trade.order, "orderId", None)
        try:
            log_event(
                "stop_submit",
                trade_id,
                arm="",
                side=child_action,
                qty=int(qty or 0),
                expected_px=stop_px,
                reason="guard_stop",
                oca=oca,
                extra={"source": "order_core", "mode": "guard_naked_position"},
            )
            _attach_fill_event_logger(
                stop_trade,
                event_name="stop_fill",
                trade_id=trade_id,
                arm="",
                side=child_action,
                qty_hint=qty,
                expected_px=stop_px,
                logger=logger,
            )
        except Exception:
            pass

        target_trade: Trade = ib.placeOrder(contract, target)
        target_id = getattr(target_trade.order, "orderId", None)
        try:
            log_event(
                "target_submit",
                trade_id,
                arm="",
                side=child_action,
                qty=int(qty or 0),
                expected_px=target_px,
                reason="guard_target",
                oca=oca,
                extra={"source": "order_core", "mode": "guard_naked_position"},
            )
            _attach_fill_event_logger(
                target_trade,
                event_name="target_fill",
                trade_id=trade_id,
                arm="",
                side=child_action,
                qty_hint=qty,
                expected_px=target_px,
                logger=logger,
            )
        except Exception:
            pass

        _record_bracket(
            logger,
            "guard_placed",
            net_qty=net_qty,
            qty=qty,
            stop_id=stop_id,
            target_id=target_id,
            stop_px=stop_px,
            target_px=target_px,
            oca=oca,
            trade_id=trade_id,
        )

    except Exception as e:
        _log(logger, "error", f"[guard_naked_position] EXCEPTION: {e}")
        _record_bracket(logger, "guard_error", net_qty=net_qty, last_px=last_px, error=str(e))


# ---------------------------------------------------------------------------
# Entry engine (moves REAL ENTRY PATH out of paper_trader.py)
# ---------------------------------------------------------------------------

def maybe_enter_trade(
    *,
    ib: IB,
    contract: Contract,
    logger,
    args,
    can_enter: bool,
    arm: Optional[str],
    side: Optional[str],  # "BUY"/"SELL"
    last_px: float,
    stop_dist: float,
    tp_dist: float,
    caps: List[str],
    net: int,
    equity: float,
    last_acct_netliq: Optional[float],
    meta_factor: float,
    hwm_factor: float,
    boost_factor: float,
    margin_mgr,
    shadow,
    last_regime: str,
    day_risk,
    es_multiplier: float = 50.0,
    short_risk_mult: float = 0.5,
) -> Tuple[List[str], bool, Optional[str], Optional[str], float]:
    """
    Moves the "REAL ENTRY PATH" out of paper_trader.
    Returns: (caps, entered_ok, new_current_arm, new_current_side, orphan_sweep_ts)
    """
    if not (can_enter and arm and side):
        return caps, False, None, None, 0.0

    s = side.upper().strip()
    if s not in ("BUY", "SELL"):
        return caps, False, None, None, 0.0

    shadow_mult, veto = shadow.entry_multiplier(regime=last_regime, arm=arm, side=s, default=1.0)
    if shadow_mult <= 0.0:
        _log(logger, "info", f"[shadow_filter] BLOCKED real entry arm={arm} side={s} shadow_mult={shadow_mult:.2f}")
        caps = (caps or []) + ([veto] if veto else ["shadow_block"])
        return caps, False, None, None, 0.0
    if veto:
        caps = (caps or []) + [veto]

    base_risk_pct = float(getattr(args, "risk_pct", 0.0) or 0.0)
    effective_risk_pct = base_risk_pct * float(meta_factor) * float(hwm_factor) * float(boost_factor) * float(shadow_mult)

    if s == "SELL":
        effective_risk_pct *= float(short_risk_mult)
        _log(logger, "info", f"[short_risk] SHORT_RISK_MULT={short_risk_mult:.2f} effective_risk_pct={effective_risk_pct:.5f}")

    equity_for_sizing = float(equity)
    if bool(getattr(args, "use_ib_pnl", False)) and (last_acct_netliq is not None):
        equity_for_sizing = float(last_acct_netliq)

    boosted_max_contracts = int(getattr(args, "max_contracts", 1) or 1)
    boosted_max_contracts = max(1, int(round(float(boosted_max_contracts) * min(float(boost_factor), 2.0))))

    contracts = dynamic_contracts(
        equity=float(equity_for_sizing),
        risk_pct=float(effective_risk_pct),
        risk_ticks=int(getattr(args, "risk_ticks", 12) or 12),
        tick_size=float(getattr(args, "tick_size", 0.25) or 0.25),
        multiplier=float(es_multiplier),
        max_contracts=int(boosted_max_contracts),
    )

    desired_delta = int(contracts) if s == "BUY" else -int(contracts)

    per_contract_init = (
        float(getattr(args, "risk_ticks", 12) or 12)
        * float(getattr(args, "tick_size", 0.25) or 0.25)
        * float(es_multiplier)
    )

    clamped_delta = margin_mgr.clamp_entry_size(
        product="ES",
        desired_qty_delta=int(desired_delta),
        current_net_qty=int(net),
        per_contract_init=float(per_contract_init),
    )
    final_qty = abs(int(clamped_delta))

    if final_qty <= 0:
        _log(logger, "warning", f"[entry] margin_core blocked entry desired_delta={desired_delta} side={s}")
        return caps, False, None, None, 0.0

    if s == "BUY":
        stop_px = float(last_px) - float(stop_dist)
        target_px = float(last_px) + float(tp_dist)
    else:
        stop_px = float(last_px) + float(stop_dist)
        target_px = float(last_px) - float(tp_dist)

    ok, parent_id, stp_id, tgt_id = place_protected_entry(
        ib=ib,
        contract=contract,
        action=s,
        qty=int(final_qty),
        stop_px=float(stop_px),
        target_px=float(target_px),
        px_hint=float(last_px),
        logger=logger,
        arm=arm,  # OPTIONAL BUT RECOMMENDED: attribution
    )

    if not ok:
        _log(logger, "error", "[entry] place_protected_entry failed; CHECK TWS.")
        return caps, False, None, None, 0.0

    new_current_arm = arm
    new_current_side = "LONG" if s == "BUY" else "SHORT"

    try:
        day_risk.last_entry_time = time.time()
    except Exception:
        pass

    orphan_ts = time.time()
    return caps, True, new_current_arm, new_current_side, float(orphan_ts)
