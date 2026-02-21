#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
backfill_trades_r.py  (v2)

Recompute R and risk_usd for existing trades.csv.

- Normalizes side: BUY/SELL -> LONG/SHORT
- Tags baseline / seed rows so they don't affect stats:
    * side blank or "?"  -> BASELINE
    * or reason in {baseline_seed, seed_equity, baseline}
- Computes risk_usd from stop_px if missing
- Computes R = pnl / risk_usd where possible
- Clamps absurd R values (|R| > 50) to blank

Output:
    results/trades_clean.csv

Then, if you're happy:
    replace trades.csv with trades_clean.csv
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional, Tuple

ES_POINT_VALUE = 50.0  # USD per ES point


def _to_float(val: str) -> Optional[float]:
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _to_int(val: str) -> Optional[int]:
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def _normalize_side(side: str) -> str:
    s = (side or "").strip().upper()
    if s in {"BUY", "LONG"}:
        return "LONG"
    if s in {"SELL", "SHORT"}:
        return "SHORT"
    if s in {"BASELINE", "SEED"}:
        return "BASELINE"
    return s  # may be "" or "?"


def _is_baseline_row(row: dict) -> bool:
    side = (row.get("side") or "").strip().upper()
    reason = (row.get("reason") or "").strip().lower()

    if side in {"", "?", "BASELINE"}:
        return True
    if reason in {"baseline_seed", "seed_equity", "baseline", "init"}:
        return True
    return False


def _compute_r_and_risk(
    qty: Optional[int],
    entry_px: Optional[float],
    stop_px: Optional[float],
    pnl: Optional[float],
    existing_risk: Optional[float],
) -> Tuple[Optional[float], Optional[float]]:
    if qty is None or entry_px is None or pnl is None:
        return None, None

    risk_usd = None

    if existing_risk is not None and existing_risk > 0:
        risk_usd = existing_risk
    elif stop_px is not None:
        pts = abs(entry_px - stop_px)
        risk_usd = pts * ES_POINT_VALUE * abs(qty)
        if risk_usd <= 0:
            risk_usd = None

    if risk_usd is None or risk_usd <= 0:
        return None, None

    R = pnl / risk_usd

    if not (-50.0 <= R <= 50.0):
        # suspicious value, keep risk but drop R
        return None, risk_usd

    return R, risk_usd


def main() -> None:
    in_path = Path("results") / "trades.csv"
    out_path = Path("results") / "trades_clean.csv"

    if not in_path.exists():
        print(f"Input file not found: {in_path}")
        return

    rows = []
    with in_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        # make sure we have expected columns
        for needed in [
            "timestamp",
            "symbol",
            "side",
            "qty",
            "entry_px",
            "exit_px",
            "stop_px",
            "target_px",
            "pnl",
            "risk_usd",
            "R",
            "strategy",
            "arm",
            "reason",
            "notes",
        ]:
            if needed not in fieldnames:
                fieldnames.append(needed)
        for row in reader:
            rows.append(row)

    cleaned = []

    for row in rows:
        # normalize side
        raw_side = row.get("side", "")
        norm_side = _normalize_side(raw_side)
        row["side"] = norm_side

        # parse common fields
        qty = _to_int(row.get("qty"))
        entry_px = _to_float(row.get("entry_px"))
        stop_px = _to_float(row.get("stop_px"))
        pnl = _to_float(row.get("pnl"))
        risk_usd_existing = _to_float(row.get("risk_usd"))

        # baseline detection
        is_baseline = _is_baseline_row(row)

        if is_baseline:
            row["side"] = "BASELINE"
            row["R"] = ""
            if not (row.get("reason") or "").strip():
                row["reason"] = "baseline_seed"
            # pnl can stay as-is (for info), but won't be used in stats
            cleaned.append(row)
            continue

        # compute risk + R
        R_val, risk_val = _compute_r_and_risk(
            qty=qty,
            entry_px=entry_px,
            stop_px=stop_px,
            pnl=pnl,
            existing_risk=risk_usd_existing,
        )

        row["risk_usd"] = f"{risk_val:.2f}" if risk_val is not None else ""
        row["R"] = f"{R_val:.6f}" if R_val is not None else ""

        # default symbol if missing
        if not (row.get("symbol") or "").strip():
            row["symbol"] = "ES"

        cleaned.append(row)

    # write cleaned file
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in cleaned:
            writer.writerow(row)

    print(f"Wrote cleaned file to {out_path}")
    print("If everything looks good, you can replace trades.csv with trades_clean.csv")


if __name__ == "__main__":
    main()

