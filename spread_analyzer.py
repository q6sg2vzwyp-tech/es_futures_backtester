#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ES Spread Analyzer v0.1

Standalone helper to analyze ES calendar spreads using IBKR historical data.

- Connects to IB via ib_core.connect_ib
- Resolves two ES contracts via ib_core.resolve_contract
- Pulls daily historical bars for each leg
- Aligns by date and computes:

    spread = front_close - back_close

- Writes CSV to results/es_spread_<FRONT>_<BACK>.csv
- Prints basic stats (mean, std, min, max, last z-score)

This is READ-ONLY: no orders, no interaction with paper_trader.
"""

import os
import sys
import csv
import math
import argparse
import datetime as dt
from typing import Dict, List, Tuple

from ib_insync import BarData  # type: ignore

import utils
from ib_core import connect_ib, resolve_contract


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ES Spread Analyzer v0.1")

    # IB connection
    p.add_argument("--ib-host", default="127.0.0.1")
    p.add_argument("--ib-port", type=int, default=4002)
    p.add_argument("--ib-client-id", type=int, default=222)

    # Shared contract settings
    p.add_argument("--exchange", default="GLOBEX")
    p.add_argument("--currency", default="USD")

    # Two legs (localSymbol style, e.g., ESZ5, ESH6)
    p.add_argument("--front-local", required=True,
                   help="Front leg localSymbol (e.g. ESZ5)")
    p.add_argument("--back-local", required=True,
                   help="Back leg localSymbol (e.g. ESH6)")

    # Historical request parameters
    p.add_argument("--duration", default="365 D",
                   help='IB duration string (e.g. "180 D", "365 D")')
    p.add_argument("--bar-size", default="1 day",
                   help='Bar size (e.g. "1 day", "1 hour")')
    p.add_argument("--what-to-show", default="TRADES")

    return p


def bars_to_dict(bars: List[BarData]) -> Dict[dt.date, float]:
    """
    Convert list of IB BarData into dict: {date -> close}
    """
    out: Dict[dt.date, float] = {}
    for b in bars:
        # b.date is str or datetime depending on formatDate; be defensive
        d_raw = getattr(b, "date", None)
        if d_raw is None:
            continue

        if isinstance(d_raw, dt.datetime):
            d = d_raw.date()
        else:
            # assume something like "20250102"
            s = str(d_raw)
            if "-" in s:
                # "YYYY-MM-DD ..."
                d = dt.datetime.fromisoformat(s.split(" ")[0]).date()
            else:
                # "YYYYMMDD"
                d = dt.datetime.strptime(s[:8], "%Y%m%d").date()

        out[d] = float(b.close)
    return out


def compute_stats(spreads: List[float]) -> Tuple[float, float, float, float]:
    """
    Return (mean, std, min, max) for a list of float spreads.
    """
    if not spreads:
        return 0.0, 0.0, 0.0, 0.0

    n = len(spreads)
    mean = sum(spreads) / n

    var = 0.0
    for x in spreads:
        var += (x - mean) ** 2
    var = var / max(1, n - 1)
    std = math.sqrt(var)

    return mean, std, min(spreads), max(spreads)


def main() -> None:
    args = build_arg_parser().parse_args()

    utils.ensure_dir(LOG_DIR)
    utils.ensure_dir(RESULTS_DIR)

    logger = utils.setup_logger(LOG_DIR, "spread_analyzer")
    logger.info("Starting ES Spread Analyzer v0.1")

    # --- Connect to IB ---
    ib = connect_ib(args, logger)

    # Build two "arg-like" objects for resolve_contract with different local symbols
    base_args = vars(args).copy()

    class _Args:
        pass

    # Front leg args
    front_arg = _Args()
    for k, v in base_args.items():
        setattr(front_arg, k, v)
    front_arg.local_symbol = args.front_local

    # Back leg args
    back_arg = _Args()
    for k, v in base_args.items():
        setattr(back_arg, k, v)
    back_arg.local_symbol = args.back_local

    # --- Resolve contracts ---
    try:
        front_con = resolve_contract(ib, front_arg, logger)
        back_con = resolve_contract(ib, back_arg, logger)
    except Exception as e:
        logger.error(f"[contracts] failed to resolve contracts: {e}")
        sys.exit(1)

    if front_con is None or back_con is None:
        logger.error("[contracts] resolve_contract returned None for one or both legs")
        sys.exit(1)

    logger.info(
        "[contracts] front=%s (conId=%s) | back=%s (conId=%s)",
        getattr(front_con, "localSymbol", front_con),
        getattr(front_con, "conId", None),
        getattr(back_con, "localSymbol", back_con),
        getattr(back_con, "conId", None),
    )

    # --- Request historical data ---
    logger.info(
        "[hist] requesting %s of %s bars: what=%s",
        args.duration,
        args.bar_size,
        args.what_to_show,
    )

    front_bars: List[BarData] = ib.reqHistoricalData(
        front_con,
        endDateTime="",
        durationStr=args.duration,
        barSizeSetting=args.bar_size,
        whatToShow=args.what_to_show,
        useRTH=False,
        formatDate=1,
        keepUpToDate=False,
    )

    back_bars: List[BarData] = ib.reqHistoricalData(
        back_con,
        endDateTime="",
        durationStr=args.duration,
        barSizeSetting=args.bar_size,
        whatToShow=args.what_to_show,
        useRTH=False,
        formatDate=1,
        keepUpToDate=False,
    )

    if not front_bars or not back_bars:
        logger.error(
            "[hist] missing data: front_bars=%d, back_bars=%d",
            len(front_bars),
            len(back_bars),
        )
        sys.exit(1)

    logger.info(
        "[hist] received %d bars (front) and %d bars (back)",
        len(front_bars),
        len(back_bars),
    )

    # --- Align by date ---
    front_dict = bars_to_dict(front_bars)
    back_dict = bars_to_dict(back_bars)

    common_dates = sorted(set(front_dict.keys()) & set(back_dict.keys()))
    if not common_dates:
        logger.error("[align] no overlapping dates between front and back legs")
        sys.exit(1)

    logger.info("[align] %d common dates for spread computation", len(common_dates))

    # --- Compute spread series ---
    rows = []
    spreads = []

    for d in common_dates:
        f_close = front_dict[d]
        b_close = back_dict[d]
        sp = f_close - b_close
        spreads.append(sp)
        rows.append((d, f_close, b_close, sp))

    mean_spread, std_spread, min_spread, max_spread = compute_stats(spreads)

    # Compute z-scores
    output_rows = []
    for d, f_close, b_close, sp in rows:
        if std_spread > 0:
            z = (sp - mean_spread) / std_spread
        else:
            z = 0.0
        output_rows.append(
            {
                "date": d.isoformat(),
                "front_close": f_close,
                "back_close": b_close,
                "spread": sp,
                "spread_mean": mean_spread,
                "spread_z": z,
            }
        )

    # --- Write CSV ---
    fname = f"es_spread_{args.front_local}_{args.back_local}.csv"
    out_path = os.path.join(RESULTS_DIR, fname)

    try:
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            fieldnames = [
                "date",
                "front_close",
                "back_close",
                "spread",
                "spread_mean",
                "spread_z",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in output_rows:
                writer.writerow(r)
        logger.info("[csv] wrote %d rows to %s", len(output_rows), out_path)
    except Exception as e:
        logger.error(f"[csv] failed to write {out_path}: {e}")
        sys.exit(1)

    # --- Print simple stats to console ---
    n = len(spreads)
    last_row = output_rows[-1]
    last_spread = last_row["spread"]
    last_z = last_row["spread_z"]

    print("")
    print("============================================")
    print(" ES Spread Analyzer v0.1")
    print("============================================")
    print(f"Front leg : {args.front_local}")
    print(f"Back leg  : {args.back_local}")
    print(f"Samples   : {n}")
    print("")
    print(f"Spread mean      : {mean_spread:.4f}")
    print(f"Spread std       : {std_spread:.4f}")
    print(f"Spread min       : {min_spread:.4f}")
    print(f"Spread max       : {max_spread:.4f}")
    print("")
    print(f"Last spread      : {last_spread:.4f}")
    print(f"Last spread z    : {last_z:.2f}")
    print("")
    print(f"CSV written to   : {out_path}")
    print("============================================")
    print("Tip: filter by |spread_z| > 1.5 or 2.0 in Excel")
    print("     to eyeball 'stretched' spread regimes.")
    print("")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Exiting...")
        sys.exit(0)

