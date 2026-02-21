#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pnl_core.py

Snapshot helpers for ES Paper Trader:

- snapshot_es_pnl_and_orders:
    * ES position avg_px and unrealized PnL (raw, position-only)
    * counts of open orders / stops / limits for the ES contract
    * lists of open order IDs (all / stops / limits)
    * best-effort stop_px / target_px guess from existing child orders
    * account-level UnrealizedPnL / RealizedPnL / NetLiquidation from IB

This is intentionally stateless so it can be called once per loop from
paper_trader.py.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from ib_insync import IB, Contract, Trade, Order


ES_MULTIPLIER = 50.0  # ES contract multiplier


def _is_stop_order(order: Order) -> bool:
    ot = (order.orderType or "").upper()
    # STP / STP LMT / TRAIL* are treated as "stop-style" protection
    if "STP" in ot:
        return True
    if ot.startswith("TRAIL"):
        return True
    return False


def _is_limit_order(order: Order) -> bool:
    ot = (order.orderType or "").upper()
    return ot == "LMT" or ot == "MIT"


def snapshot_es_pnl_and_orders(
    *,
    ib: IB,
    con: Contract,
    last_px: float,
    logger,
) -> Tuple[
    Optional[float],  # es_avg_px
    Optional[float],  # es_unreal_pnl_raw
    int,              # es_open_orders
    int,              # es_open_stops
    int,              # es_open_limits
    List[int],        # open_order_ids
    List[int],        # open_stop_ids
    List[int],        # open_limit_ids
    Optional[float],  # stop_px
    Optional[float],  # target_px
    Optional[float],  # acct_unreal_pnl
    Optional[float],  # acct_realized_pnl
    Optional[float],  # acct_netliq
]:
    """
    Take a one-shot snapshot of ES position, orders, and account PnL.

    - es_avg_px:    avgCost for ES position (if any)
    - es_unreal_pnl_raw: (last_px - es_avg_px) * ES_MULTIPLIER * net_qty
    - es_open_orders: number of open trades for this ES contract
    - es_open_stops / es_open_limits: classified by orderType
    - stop_px / target_px: best-effort guess from child orders
    - acct_unreal_pnl / acct_realized_pnl / acct_netliq: from account summary
    """

    es_avg_px: Optional[float] = None
    es_unreal_pnl_raw: Optional[float] = None

    es_open_orders: int = 0
    es_open_stops: int = 0
    es_open_limits: int = 0

    open_order_ids: List[int] = []
    open_stop_ids: List[int] = []
    open_limit_ids: List[int] = []

    stop_px: Optional[float] = None
    target_px: Optional[float] = None

    acct_unreal_pnl: Optional[float] = None
    acct_realized_pnl: Optional[float] = None
    acct_netliq: Optional[float] = None

    # -------------------- Position snapshot (ES only) --------------------
    net_qty = 0
    try:
        for pos in ib.positions():
            try:
                if getattr(pos.contract, "conId", None) != getattr(con, "conId", None):
                    continue
            except Exception:
                continue

            net_qty = int(getattr(pos, "position", 0))
            es_avg_px = float(getattr(pos, "avgCost", 0.0) or 0.0)

            if last_px is not None and net_qty != 0:
                es_unreal_pnl_raw = (float(last_px) - es_avg_px) * ES_MULTIPLIER * net_qty
            break
    except Exception as e:
        logger.error(f"[pnl_core] error while reading positions: {e}")

    # -------------------- Open orders for this ES contract ----------------
    try:
        for trade in ib.openTrades():
            t_con = getattr(trade, "contract", None)
            if t_con is None:
                continue
            if getattr(t_con, "conId", None) != getattr(con, "conId", None):
                continue

            order = trade.order
            oid = int(getattr(order, "orderId", -1))
            es_open_orders += 1
            if oid >= 0:
                open_order_ids.append(oid)

            if _is_stop_order(order):
                es_open_stops += 1
                if oid >= 0:
                    open_stop_ids.append(oid)
                # For first stop, latch price (auxPrice or lmtPrice fallback)
                if stop_px is None:
                    px = None
                    if getattr(order, "auxPrice", 0.0):
                        px = float(order.auxPrice)
                    elif getattr(order, "lmtPrice", 0.0):
                        px = float(order.lmtPrice)
                    stop_px = px
            elif _is_limit_order(order):
                es_open_limits += 1
                if oid >= 0:
                    open_limit_ids.append(oid)
                # For first limit, latch lmtPrice if present
                if target_px is None and getattr(order, "lmtPrice", 0.0):
                    target_px = float(order.lmtPrice)
    except Exception as e:
        logger.error(f"[pnl_core] error while scanning openTrades: {e}")

    # -------------------- Account-level PnL / NetLiq ---------------------
    try:
        # accountSummary() returns a list of objects with (tag, value)
        summary = ib.accountSummary()
        for v in summary:
            tag = getattr(v, "tag", "")
            val = getattr(v, "value", None)
            if val is None:
                continue
            try:
                fval = float(val)
            except Exception:
                continue

            if tag == "UnrealizedPnL":
                acct_unreal_pnl = fval
            elif tag == "RealizedPnL":
                acct_realized_pnl = fval
            elif tag == "NetLiquidation":
                acct_netliq = fval
    except Exception as e:
        logger.error(f"[pnl_core] error while reading accountSummary: {e}")

    return (
        es_avg_px,
        es_unreal_pnl_raw,
        es_open_orders,
        es_open_stops,
        es_open_limits,
        open_order_ids,
        open_stop_ids,
        open_limit_ids,
        stop_px,
        target_px,
        acct_unreal_pnl,
        acct_realized_pnl,
        acct_netliq,
    )

