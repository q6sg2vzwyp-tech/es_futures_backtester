#!/usr/bin/env python
# test_bracket_plumbing.py (hardened)
# Verifies bracket order plumbing with robust price discovery and fallbacks.

import argparse
import math
import sys
import time

from ib_insync import IB, Contract, MarketOrder, Order


def finite(x):
    try:
        return (x is not None) and math.isfinite(float(x)) and float(x) > 0
    except Exception:
        return False


def poll_market_price(ib: IB, contract: Contract, timeout=5.0):
    """Try to get a reasonable price within timeout."""
    md = ib.reqMktData(contract, "", False, False)
    deadline = time.time() + float(timeout)
    px = None
    while time.time() < deadline:
        last = md.last
        close = md.close
        bid = md.bid
        ask = md.ask
        mid = None
        if finite(bid) and finite(ask):
            mid = (bid + ask) / 2.0

        for cand in (last, close, mid, ask, bid):
            if finite(cand):
                px = float(cand)
                break
        if px:
            break
        ib.sleep(0.2)

    ib.cancelMktData(contract)
    return px


def round2(x):
    return float(f"{x:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7497)
    ap.add_argument("--clientId", type=int, default=99)
    ap.add_argument("--conId", type=int, required=True)
    ap.add_argument("--qty", type=int, default=1)
    ap.add_argument("--risk-ticks", type=int, default=8, help="Defines R; ES tick=0.25")
    ap.add_argument("--tick-size", type=float, default=0.25)
    ap.add_argument("--side", choices=["LONG", "SHORT"], default="LONG")
    ap.add_argument("--tp-R", type=float, default=1.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ib = IB()
    print(f"[TEST] Connecting {args.host}:{args.port} clientId={args.clientId}")
    ib.connect(args.host, args.port, clientId=args.clientId)
    print("[TEST] Connected.")

    # Resolve contract from conId
    c = Contract()
    c.secType = "FUT"
    c.exchange = "CME"
    c.conId = int(args.conId)
    cds = ib.reqContractDetails(c)
    if not cds:
        print("[TEST] ERROR: No contract details for conId", args.conId)
        sys.exit(2)
    contract = cds[0].contract
    print(f"[TEST] Using {contract.localSymbol or contract.symbol} (conId={contract.conId})")

    # Attempt market price (for preview only; children will also fall back to avgFill)
    snap_px = poll_market_price(ib, contract, timeout=5.0)
    print(f"[TEST] Snap price ~ {snap_px if snap_px else 'N/A'}")
    R = float(args.risk_ticks) * float(args.tick_size)

    if args.side == "LONG":
        parent_action, child_action = "BUY", "SELL"
    else:
        parent_action, child_action = "SELL", "BUY"

    print(f"[TEST] Plan: {args.side} {args.qty} @ MKT  |  R={R:.2f}  (TP {args.tp_R}R)")
    if args.dry_run:
        print("[TEST] Dry-run: no orders sent.")
        ib.disconnect()
        return

    # Parent market order (not transmitted yet)
    parent = MarketOrder(parent_action, int(args.qty))
    parent.transmit = False
    parent.orderRef = "PLUMBING-TEST"
    trade = ib.placeOrder(contract, parent)

    # Wait until parent is working or filled
    for _ in range(60):  # up to ~15s
        st = trade.orderStatus.status
        if st in ("Submitted", "PreSubmitted", "Filled"):
            break
        ib.sleep(0.25)

    parentId = trade.order.orderId
    avg = float(trade.orderStatus.avgFillPrice or 0.0)

    # Decide base price for children:
    # 1) use avgFill if available,
    # 2) else use snap price if available,
    # 3) else wait briefly once more for avgFill,
    # 4) else cancel parent and exit (to avoid NaN).
    base = avg if finite(avg) else (snap_px if finite(snap_px) else None)

    if not finite(base):
        # one last short wait in case market order filled just now
        for _ in range(20):  # ~5s
            avg = float(trade.orderStatus.avgFillPrice or 0.0)
            if finite(avg):
                base = avg
                break
            ib.sleep(0.25)

    if not finite(base):
        print(
            "[TEST] ERROR: No valid price for bracket children (avgFill and snapshot both missing)."
        )
        print("[TEST] Cancelling parent to be safe.")
        try:
            ib.cancelOrder(trade.order)
        except Exception:
            pass
        ib.disconnect()
        sys.exit(3)

    # Build stop/tp from base
    if args.side == "LONG":
        stop_px = round2(base - R)
        tp_px = round2(base + args.tp_R * R)
    else:
        stop_px = round2(base + R)
        tp_px = round2(base - args.tp_R * R)

    print(f"[TEST] Using base={base:.2f} -> TP={tp_px:.2f}  STP={stop_px:.2f}")

    stop = Order()
    stop.action = child_action
    stop.orderType = "STP"
    stop.auxPrice = float(stop_px)
    stop.totalQuantity = int(args.qty)
    stop.parentId = parentId
    stop.transmit = False

    tp = Order()
    tp.action = child_action
    tp.orderType = "LMT"
    tp.lmtPrice = float(tp_px)
    tp.totalQuantity = int(args.qty)
    tp.parentId = parentId
    tp.transmit = True  # last child transmits all

    # Place children
    ib.placeOrder(contract, stop)
    ib.placeOrder(contract, tp)

    print(f"[TEST] Parent orderId={parentId} placed; children submitted.")
    print(
        "[TEST] Check TWS: you should see a parent + LMT TP + STP stop bracket.\n"
        "      Cancel/manage in TWS when done. Ctrl+C here to exit."
    )

    try:
        while True:
            ib.sleep(1.0)
    except KeyboardInterrupt:
        print("[TEST] Exiting.")
    finally:
        ib.disconnect()


if __name__ == "__main__":
    main()
