#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from ib_insync import IB, Future


def main() -> int:
    ib = IB()
    print("Connecting to IB...")
    ib.connect("127.0.0.1", 4002, clientId=999)

    candidates = [
        Future(symbol="ES", exchange="GLOBEX", currency="USD"),
        Future(symbol="ES", exchange="CME",    currency="USD"),
    ]

    try:
        for base in candidates:
            print(f"\n=== Trying {base.symbol} {base.exchange} {base.currency} ===")
            cds = ib.reqContractDetails(base)
            print(f"  contractDetails count: {len(cds)}")
            for cd in cds[:5]:
                c = cd.contract
                print(f"   -> {c.localSymbol}  {c.lastTradeDateOrContractMonth}")
    finally:
        try:
            ib.disconnect()
        except Exception:
            pass

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
