from __future__ import annotations

import time
from typing import Callable, Optional, Any, Dict

try:
    from ib_insync import MarketOrder  # type: ignore
except Exception:  # pragma: no cover
    MarketOrder = None  # type: ignore


def flatten_market(ib, con, args, log: Callable[..., Any], risk, net_qty: int) -> None:
    """Flatten position with a market order. Side-effecting. Mirrors prior inline logic."""
    if not getattr(args, "place_orders", False):
        return
    side = "SELL" if net_qty > 0 else "BUY"
    qty = abs(int(net_qty))
    try:
        if MarketOrder is None:
            raise RuntimeError("ib_insync.MarketOrder unavailable")
        mo = MarketOrder(side, qty)
        ib.placeOrder(con, mo)
        log("flatten_market", side=side, qty=qty)
        try:
            risk.last_flat_fill_ts = time.time()
        except Exception:
            pass
    except Exception as e:
        log("flatten_err", err=str(e))


def promote_parent_limit_to_market(
    *,
    ib,
    con,
    args,
    log: Callable[..., Any],
    hb_update: Callable[..., Any],
    parent_entry_id: Optional[int],
    parent_entry_time: Optional[float],
    parent_promote_deadline: Optional[float],
    parent_missing_log_next: float,
    net_qty: int,
    ACTIVE_STATUSES,
    is_our_order_or_trade: Callable[..., bool],
    safe_cancel: Callable[..., bool],
    reset_parent_tracking: Callable[[], None],
) -> float:
    """Auto-promote a stale parent LIMIT entry to MARKET (or fallback market) if it sits too long.

    Returns updated parent_missing_log_next (other parent tracking is reset via reset_parent_tracking()).
    """
    if not (getattr(args, "parent_to_mkt_sec", 0) > 0 and parent_entry_id is not None and net_qty == 0):
        return parent_missing_log_next

    try:
        now_ts = time.time()
        limit_sec = max(1, int(getattr(args, "parent_to_mkt_sec", 0)))
        age = max(0.0, now_ts - (parent_entry_time or now_ts))
        if parent_promote_deadline is not None:
            remaining = max(0, int(parent_promote_deadline - now_ts))
        else:
            remaining = max(0, int(limit_sec - age))

        hb_update(
            parent_entry_id=parent_entry_id,
            parent_to_mkt_limit_sec=limit_sec,
            parent_to_mkt_age_sec=int(age),
            parent_to_mkt_remaining_sec=remaining,
        )

        if age < limit_sec:
            return parent_missing_log_next

        found_working = False
        should_reset_parent = False

        for t in ib.openTrades():
            if not is_our_order_or_trade(t):
                continue
            if getattr(t.order, "orderId", None) != parent_entry_id:
                continue

            st = (getattr(t.orderStatus, "status", "") or "").strip()
            if st not in ACTIVE_STATUSES:
                should_reset_parent = True
                log("parent_to_market_inactive", orderId=parent_entry_id, status=st, age=int(age))
                break

            found_working = True
            o = t.order
            from_price = getattr(o, "lmtPrice", None)

            # Flip the parent to MARKET
            try:
                o.orderType = "MKT"
            except Exception:
                pass
            try:
                o.lmtPrice = 0.0
            except Exception:
                pass

            try:
                trade = ib.placeOrder(con, o)  # modify existing order
                ib_status = getattr(getattr(trade, "orderStatus", None), "status", None) if trade is not None else None
                log(
                    "parent_to_market",
                    orderId=getattr(o, "orderId", None),
                    time_in_state_sec=int(age),
                    from_price=from_price,
                    side=getattr(o, "action", None),
                    ib_status=ib_status,
                )
                should_reset_parent = True
            except Exception as e:
                log(
                    "parent_to_market_modify_err",
                    err=str(e),
                    orderId=getattr(o, "orderId", None),
                    time_in_state_sec=int(age),
                    from_price=from_price,
                    side=getattr(o, "action", None),
                )

                qty = abs(int(getattr(o, "totalQuantity", 0) or 0)) or 1
                try:
                    if MarketOrder is None:
                        raise RuntimeError("ib_insync.MarketOrder unavailable")
                    fallback = MarketOrder(getattr(o, "action", None), qty)
                    trade_fb = ib.placeOrder(con, fallback)
                    fb_status = getattr(getattr(trade_fb, "orderStatus", None), "status", None) if trade_fb is not None else None
                    log(
                        "parent_to_market_fallback",
                        parent_order=getattr(o, "orderId", None),
                        fallback_order_id=(
                            getattr(fallback, "orderId", None)
                            or getattr(getattr(trade_fb, "order", None), "orderId", None)
                        ),
                        side=getattr(o, "action", None),
                        qty=qty,
                        ib_status=fb_status,
                    )

                    parent_cancelled = False
                    parent_status = (getattr(t.orderStatus, "status", "") or "").strip()
                    try:
                        if parent_status not in ACTIVE_STATUSES:
                            parent_cancelled = True
                            log("parent_to_market_fallback_parent_inactive", orderId=getattr(o, "orderId", None), status=parent_status)
                        else:
                            parent_cancelled = bool(safe_cancel(ib, t, note="[parent_to_market_fallback]"))
                            if parent_cancelled:
                                log("parent_to_market_fallback_parent_cancelled", orderId=getattr(o, "orderId", None), status=parent_status)
                    except Exception as cancel_err:
                        log("parent_to_market_fallback_parent_cancel_err", orderId=getattr(o, "orderId", None), err=str(cancel_err), status=parent_status)

                    if parent_cancelled:
                        should_reset_parent = True
                    else:
                        log("parent_to_market_fallback_parent_manual_cleanup", orderId=getattr(o, "orderId", None), status=parent_status)

                except Exception as inner:
                    log("parent_to_market_fallback_err", err=str(inner), side=getattr(o, "action", None), qty=qty)

            break  # we handled the parent we found

        if should_reset_parent:
            reset_parent_tracking()
        elif not found_working:
            now_ts = time.time()
            if now_ts >= parent_missing_log_next:
                log("parent_to_market_not_found", age=int(age), orderId=parent_entry_id)
                parent_missing_log_next = now_ts + 5.0

    except Exception as e:
        log("parent_to_market_block_err", err=str(e), orderId=parent_entry_id)

    return parent_missing_log_next


def place_bracket(ctx: Dict[str, Any], state: Dict[str, Any], go_long: bool, last_price: float, last_bar_ts_local, net_qty_now: int, signal_name: Optional[str]=None) -> None:
    """Extracted from paper_trader.place_bracket (nested). Mutates `state` in-place."""
    LimitOrder = ctx['LimitOrder']
    StopOrder = ctx['StopOrder']
    args = ctx['args']
    cadence_scale = ctx['cadence_scale']
    con = ctx['con']
    ct_now = ctx['ct_now']
    current_arm = ctx['current_arm']
    equity = ctx['equity']
    equity_hwm = ctx['equity_hwm']
    has_active_parent_entry = ctx['has_active_parent_entry']
    hb_update = ctx['hb_update']
    ib = ctx['ib']
    ib_netliq = ctx['ib_netliq']
    ib_position_truth = ctx['ib_position_truth']
    log = ctx['log']
    param_arms = ctx['param_arms']
    param_learner = ctx['param_learner']
    px_mult = ctx['px_mult']
    risk = ctx['risk']
    round_to_tick = ctx['round_to_tick']
    sizing_determine_order_qty = ctx['sizing_determine_order_qty']
    ticks_to_price_delta = ctx['ticks_to_price_delta']
    persist_fn = ctx['persist_fn']
    entry_price = state.get('entry_price')
    last_entry_bar_ts = state.get('last_entry_bar_ts')
    current_param_arm = state.get('current_param_arm')
    cycle_risk_ticks = state.get('cycle_risk_ticks')
    cycle_bracket_entry_price = state.get('cycle_bracket_entry_price')
    cycle_bracket_stop_dist = state.get('cycle_bracket_stop_dist')
    cycle_bracket_tp_dist = state.get('cycle_bracket_tp_dist')
    parent_entry_id = state.get('parent_entry_id')
    parent_entry_time = state.get('parent_entry_time')
    last_attempt_ts = state.get('last_attempt_ts')

    # ---- hard anti-burst guard (attempt spacing) ----
    now_ts = time.time()
    eff_min_gap = int(max(1, args.min_seconds_between_entries * cadence_scale))
    entry_signal = signal_name or current_arm or ("long" if go_long else "short")

    def _entry_log(event: str, **fields):
        base = {
            "signal": entry_signal,
            "cadence_scale": round(cadence_scale, 3),
            "throttle_min_gap": eff_min_gap,
            "last_attempt_ts": last_attempt_ts,
        }
        base.update(fields)
        log(event, **base)

    if last_attempt_ts is not None and (now_ts - last_attempt_ts) < eff_min_gap:
        _entry_log(
            "gate_skip",
            reason="attempt_gap",
            since_last=round(now_ts - last_attempt_ts, 3),
        )
        return
    last_attempt_ts = now_ts

    if not args.place_orders:
        _entry_log("sim_no_place", reason="--place-orders not set")
        return

    # ---- trust IB position truth: only open if truly flat ----
    qty_truth, _ = ib_position_truth(ib, con)
    if qty_truth != 0:
        _entry_log(
            "gate_skip",
            reason="non_flat_ib_truth",
            qty_truth=int(qty_truth),
            net_qty_now=int(net_qty_now),
        )
        return

    if has_active_parent_entry(ib, con):
        _entry_log("gate_skip", reason="active_parent_entry")
        return
    if (
        args.debounce_one_bar
        and last_entry_bar_ts is not None
        and last_bar_ts_local == last_entry_bar_ts
    ):
        _entry_log("gate_skip", reason="debounce_one_bar")
        return


    # ---- parameter meta-learning override (per-entry) ----
    chosen_params: Dict[str, float] = {}
    if param_learner:
        arm, _ = param_learner.choose(list(param_arms.keys()), sample=(args.learn_mode != "shadow"))
        chosen_params = param_arms.get(arm, {})
        current_param_arm = arm
        risk_ticks_local = int(chosen_params.get("risk_ticks", args.risk_ticks))
        tp_R_local = float(chosen_params.get("tp_R", args.tp_R))
        entry_slip_local = int(chosen_params.get("entry_slippage_ticks", args.entry_slippage_ticks))
    else:
        current_param_arm = None
        risk_ticks_local = args.risk_ticks
        tp_R_local = args.tp_R
        entry_slip_local = args.entry_slippage_ticks

    qty = sizing_determine_order_qty(current_net_qty=net_qty_now, risk_ticks_for_trade=risk_ticks_local, args=args, px_mult=px_mult, equity=equity, equity_hwm=equity_hwm, ib_netliq=ib_netliq)
    if qty <= 0:
        _entry_log("gate_skip", reason="qty_le_0", suggested_qty=qty)
        return

    tick = float(args.tick_size)
    slippage = ticks_to_price_delta(risk_ticks_local * 0 + entry_slip_local, tick)  # slippage based on entry_slip_local
    risk_px = ticks_to_price_delta(risk_ticks_local, tick)

    if go_long:
        entry = round_to_tick(last_price + slippage, tick)
        sl = round_to_tick(entry - risk_px, tick)
        tp = round_to_tick(entry + risk_px * float(tp_R_local), tick)
        action = "BUY"
    else:
        entry = round_to_tick(last_price - slippage, tick)
        sl = round_to_tick(entry + risk_px, tick)
        tp = round_to_tick(entry - risk_px * float(tp_R_local), tick)
        action = "SELL"

    # --- parent entry ---
    parent = LimitOrder(
        action=action,
        totalQuantity=qty,
        lmtPrice=entry,
        tif=args.tif,
        outsideRth=bool(args.outsideRth),
    )
    parent.transmit = False

    try:
        ib.placeOrder(con, parent)
        parent_id = parent.orderId
        _entry_log(
            "entry_parent_order",
            order_id=parent_id,
            side=action,
            qty=qty,
            price=entry,
            transmit=parent.transmit,
            tif=args.tif,
        )
    except Exception as e:
        _entry_log("bracket_parent_error", error=str(e), side=action, qty=qty, price=entry)
        return

    ib.sleep(0.02)  # let TWS register the parent

    exit_action = "SELL" if action == "BUY" else "BUY"
    oca = f"OCA-{int(time.time()*1000)}"

    # --- stop-loss leg ---
    stop_loss = StopOrder(
        action=exit_action,
        totalQuantity=qty,
        stopPrice=sl,
        tif=args.tif,
        outsideRth=bool(args.outsideRth),
    )
    try:
        stop_loss.triggerMethod = 2
    except Exception:
        pass
    try:
        stop_loss.parentId = parent_id
        stop_loss.ocaGroup = oca
        stop_loss.transmit = False
        stop_loss.ocaType = 1
    except Exception:
        pass

    # --- take-profit leg ---
    take_profit = LimitOrder(
        action=exit_action,
        totalQuantity=qty,
        lmtPrice=tp,
        tif=args.tif,
        outsideRth=bool(args.outsideRth),
    )
    try:
        take_profit.parentId = parent_id
        take_profit.ocaGroup = oca
        take_profit.transmit = True  # transmit whole bracket
        take_profit.ocaType = 1
    except Exception:
        pass

    # Place children
    stop_order_id = None
    try:
        ib.placeOrder(con, stop_loss)
        stop_order_id = stop_loss.orderId
        _entry_log(
            "entry_child_order",
            child="stop",
            order_id=stop_order_id,
            parent_id=parent_id,
            price=sl,
            qty=qty,
            transmit=stop_loss.transmit,
            oca_group=oca,
        )
    except Exception as e:
        _entry_log(
            "bracket_children_error",
            child="stop",
            error=str(e),
            parent_id=parent_id,
            price=sl,
            qty=qty,
        )
        return

    tp_order_id = None
    try:
        ib.placeOrder(con, take_profit)
        tp_order_id = take_profit.orderId
        _entry_log(
            "entry_child_order",
            child="take_profit",
            order_id=tp_order_id,
            parent_id=parent_id,
            price=tp,
            qty=qty,
            transmit=take_profit.transmit,
            oca_group=oca,
        )
    except Exception as e:
        _entry_log(
            "bracket_children_error",
            child="take_profit",
            error=str(e),
            parent_id=parent_id,
            price=tp,
            qty=qty,
            stop_order_id=stop_order_id,
        )
        return

    _entry_log(
        "bracket_submitted",
        side=action,
        qty=qty,
        entry=entry,
        stop=sl,
        tp=tp,
        param_arm=current_param_arm,
        params=chosen_params or None,
        parent_id=parent_id,
        stop_order_id=stop_order_id,
        tp_order_id=tp_order_id,
        oca_group=oca,
    )

    # track this parent LIMIT entry for possible limit→market promotion
    parent_entry_id = parent_id
    parent_entry_time = time.time()
    parent_promote_deadline = (
        parent_entry_time + max(1, int(args.parent_to_mkt_sec))
        if args.parent_to_mkt_sec > 0
        else None
    )

    if args.parent_to_mkt_sec > 0:
        limit_sec = max(1, int(args.parent_to_mkt_sec))
        hb_update(
            parent_entry_id=parent_entry_id,
            parent_to_mkt_limit_sec=limit_sec,
            parent_to_mkt_age_sec=0,
            parent_to_mkt_remaining_sec=limit_sec,
        )
        log(
            "parent_to_market_timer_start",
            orderId=parent_entry_id,
            limit_sec=limit_sec,
        )
    else:
        hb_update(
            parent_entry_id=parent_entry_id,
            parent_to_mkt_limit_sec=None,
            parent_to_mkt_age_sec=None,
            parent_to_mkt_remaining_sec=None,
        )

    # update local state
    entry_price = entry
    cycle_bracket_entry_price = entry
    cycle_bracket_stop_dist = abs(entry - sl)
    cycle_bracket_tp_dist = abs(tp - entry)
    last_entry_bar_ts = last_bar_ts_local
    risk.last_entry_time = time.time()
    eff_cool = int(max(1, args.strategy_cooldown_sec * cadence_scale))
    risk.cool_until = ct_now() + dt.timedelta(seconds=eff_cool)
    risk.trades += 1
    cycle_risk_ticks = risk_ticks_local  # NEW: remember ticks for this cycle
    persist_fn()

    state['entry_price'] = entry_price
    state['last_entry_bar_ts'] = last_entry_bar_ts
    state['current_param_arm'] = current_param_arm
    state['cycle_risk_ticks'] = cycle_risk_ticks
    state['cycle_bracket_entry_price'] = cycle_bracket_entry_price
    state['cycle_bracket_stop_dist'] = cycle_bracket_stop_dist
    state['cycle_bracket_tp_dist'] = cycle_bracket_tp_dist
    state['parent_entry_id'] = parent_entry_id
    state['parent_entry_time'] = parent_entry_time
    state['last_attempt_ts'] = last_attempt_ts
