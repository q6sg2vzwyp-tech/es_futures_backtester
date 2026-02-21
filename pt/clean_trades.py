#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
clean_trades.py

Normalize existing trades.csv:

- Ensures a proper header
- Normalizes fields
- Computes R where missing
- Rewrites the file, keeping a .bak
"""

import argparse
from trades_core import (
    load_trades_safe,
    write_trades,
    DEFAULT_RISK_TICKS,
    DEFAULT_TICK_VALUE,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trades",
        default=r"C:\Users\owner\Desktop\es_futures_backtester\results\trades.csv",
        help="Path to trades.csv",
    )
    parser.add_argument(
        "--risk-ticks",
        type=float,
        default=DEFAULT_RISK_TICKS,
        help="Per-trade risk in ticks (for R computation)",
    )
    parser.add_argument(
        "--tick-value",
        type=float,
        default=DEFAULT_TICK_VALUE,
        help="Tick value in USD (for R computation)",
    )
    args = parser.parse_args()

    trades = load_trades_safe(
        args.trades,
        risk_ticks=args.risk_ticks,
        tick_value=args.tick_value,
    )

    print(f"Loaded {len(trades)} trades from {args.trades}")

    write_trades(args.trades, trades, backup=True)

    print(f"Normalized trades written back to {args.trades}")
    print(f"Backup created: {args.trades}.bak")


if __name__ == "__main__":
    main()

