#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
inspect_positions.py

Quick status check for your IB paper account:

- Connects to IB Gateway (127.0.0.1:4002)
- Prints ES positions (ES futures only)
- Prints open STOP/LIMIT orders (stops + profit targets)
- Shows a rough unrealized PnL for ES (per-position and total)

This does NOT place or cancel any orders.
"""

import time
from typing import List

from ib_insync import IB, Contract, Position, Trade, Ticker


HOST = "127.0.0.1"
PORT = 4002          # IB Gateway paper port
CLIENT_ID = 999      # separate from your algo (which uses 111)


def connect_ib() -> IB:
    ib = IB()
    print(f"Connecting to IB {HOST}:{PORT} clientId={CLIENT_ID} ...")
    ib.connect(HOST, PORT, clientId=CLIENT_ID)
    print("Connected.")
    return ib


def is_es_future(contract: Contract) -> bool:
    try:
        return (
            contract.secType == "FUT"
            and (contract.symbol or "") == "ES"
        )
    except Exception:
        return False


def snapshot_es_positions(ib: IB):
    """
    Return list of ES futures positions and a mapping from conId -> contract.
    """
    positions: List[Position] = ib.positions()
    es_positions = [p for p in positions if is_es_future(p.contract)]

    con_by_id = {p.contract.conId: p.contract for p in es_positions}
    return es_positions, con_by_id


def load_market_data(ib: IB, contracts: List[Contract]):
    """
    Request simple snapshot market data for a list of contracts.
    Returns dict conId -> last_price (or None).
    """
    prices = {}
    if not contracts:
        return prices

    tickers: List[Ticker] = []
    for con in contracts:
        t = ib.reqMktData(con, "", False, False)
        tickers.append(t)

    # give IB a moment to deliver data
    ib.sleep(1.0)

    for con, t in zip(contracts, tickers):
        last = t.last or t.marketPrice()
        if last is None:
            prices[con.conId] = None
        else:
            prices[con.conId] = float(last)

    # optional: cancel subscriptions to be polite
    for t in tickers:
        ib.cancelMktData(t)

    return prices


def print_positions_and_pnl(ib: IB):
    es_positions, con_by_id = snapshot_es_positions(ib)

    if not es_positions:
        print("\n=== ES Positions ===")
        print("No ES futures positions found.")
        return

    # Get market prices for those contracts
    contracts = list(con_by_id.values())
    prices = load_market_data(ib, contracts)

    print("\n=== ES Positions (Futures) ===")
    total_unreal_pnl = 0.0

    for p in es_positions:
        con = p.contract
        qty = p.position
        avg_cost = float(p.avgCost or 0.0)
        last_px = prices.get(con.conId)

        side = "LONG" if qty > 0 else "SHORT"
        mult = float(getattr(con, "multiplier", 50) or 50)

        unreal = None
        if last_px is not None:
            # For LONG: (last - avg) * qty * multiplier
            # For SHORT: (avg - last) * |qty| * multiplier
            if qty > 0:
                unreal = (last_px - avg_cost) * qty * mult
            elif qty < 0:
                unreal = (avg_cost - last_px) * abs(qty) * mult

        if unreal is not None:
            total_unreal_pnl += unreal

        print(f"- {side} {qty} x {con.localSymbol}  @ avg {avg_cost:.2f}")
        print(f"  conId={con.conId}, exch={con.exchange}, mult={mult}")
        if last_px is not None:
            print(f"  last={last_px:.2f}")
        else:
            print("  last=NA (no market price yet)")

        if unreal is not None:
            print(f"  unrealized PnL â‰ˆ {unreal:.2f} USD")
        print()

    print(f"Total unrealized ES PnL â‰ˆ {total_unreal_pnl:.2f} USD\n")


def print_open_stops_and_targets(ib: IB):
    """
    Print all open STOP/LIMIT orders for ES futures (your stops & profit takers).
    """
    trades: List[Trade] = ib.openTrades()
    es_trades = [t for t in trades if is_es_future(t.contract)]

    print("=== ES Protective Orders (Stops / Targets / Other Open Orders) ===")
    if not es_trades:
        print("No open ES futures orders.")
        return

    for t in es_trades:
        con = t.contract
        o = t.order
        status = (t.orderStatus.status or "").upper()
        ot = (o.orderType or "").upper()
        action = (o.action or "").upper()
        qty = o.totalQuantity

        # Focus on STOP/LIMIT orders (hedges)
        print(f"- orderId={o.orderId} | {status}")
        print(f"  {action} {qty} x {con.localSymbol} | type={ot}")

        # Price fields
        if hasattr(o, "lmtPrice") and o.lmtPrice:
            print(f"  limit price = {o.lmtPrice}")
        if hasattr(o, "auxPrice") and o.auxPrice:
            print(f"  aux/stop price = {o.auxPrice}")

        # OCA group info (brackets)
        ocag = (o.ocaGroup or "").strip()
        if ocag:
            print(f"  OCA group = {ocag}")

        print()


def main():
    ib = connect_ib()
    try:
        print_positions_and_pnl(ib)
        print_open_stops_and_targets(ib)
    finally:
        ib.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()

