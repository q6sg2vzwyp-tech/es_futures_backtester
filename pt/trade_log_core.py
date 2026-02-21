#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
trade_log_core.py  (v2.2 - Path-safe)

Centralized trade logging for ES Paper Trader.

Fixes:
- trades_path may be passed as a string from callers; normalize to pathlib.Path
  so .exists(), .open(), .parent work reliably.

Keeps:
- Appends a single CSV: results/trades.csv
- Ensures header is present.
- Computes R per trade (pnl / dollar_risk).
- Normalizes side: BUY/SELL -> LONG/SHORT.
- Plays nice with pnl_core / trade_bridge.

For ES:
- Tick size 0.25
- Multiplier 50 USD/point
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

ES_POINT_VALUE = 50.0  # USD per ES point (1.0)

PathLike = Union[str, Path]


@dataclass
class TradeRecord:
    """
    A single *closed* trade.

    REQUIRED:
        timestamp  : ISO-8601 string "YYYY-MM-DDTHH:MM:SS"
        side       : "LONG" or "SHORT" or "BASELINE"
        qty        : integer
        entry_px   : float
        exit_px    : float
        pnl        : float (USD)
        R          : string ("" if unknown)
        reason     : close reason / tag

    OPTIONAL:
        symbol     : "ES"
        strategy   : strategy name (e.g. "trend")
        arm        : learner arm label (e.g. "trend_ema")
        stop_px    : stop used for risk calc
        target_px  : target level (debug)
        risk_usd   : dollar risk used to compute R
        notes      : free-form text
    """

    timestamp: str
    side: str
    qty: int
    entry_px: float
    exit_px: float
    pnl: float
    R: str
    reason: str

    symbol: str = "ES"
    strategy: str = ""
    arm: str = ""
    stop_px: Optional[float] = None
    target_px: Optional[float] = None
    risk_usd: Optional[float] = None
    notes: str = ""


def _as_path(p: Optional[PathLike]) -> Optional[Path]:
    if p is None:
        return None
    if isinstance(p, Path):
        return p
    return Path(str(p))


def _default_trades_path() -> Path:
    return Path("results") / "trades.csv"


def _ensure_parent(path: Path) -> None:
    # Path-safe; idempotent.
    path.parent.mkdir(parents=True, exist_ok=True)


def _file_has_header(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8") as f:
            return bool(f.readline())
    except FileNotFoundError:
        return False


def _write_header(path: Path) -> None:
    _ensure_parent(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
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
            ]
        )


def _normalize_side(side: str) -> str:
    s = (side or "").strip().upper()
    if s in {"BUY", "LONG"}:
        return "LONG"
    if s in {"SELL", "SHORT"}:
        return "SHORT"
    if s in {"BASELINE", "SEED"}:
        return "BASELINE"
    return "?"


def _compute_r_from_inputs(
    side: str,
    qty: int,
    entry_px: float,
    stop_px: Optional[float],
    pnl: float,
    explicit_risk_usd: Optional[float],
) -> Tuple[Optional[float], Optional[float]]:
    """
    Returns (R, risk_usd).

    Priority:
      1) explicit_risk_usd if > 0
      2) abs(entry - stop) * ES_POINT_VALUE * abs(qty)
    """
    risk_usd = None

    if explicit_risk_usd is not None and explicit_risk_usd > 0:
        risk_usd = float(explicit_risk_usd)
    elif stop_px is not None:
        pts = abs(float(entry_px) - float(stop_px))
        risk_usd = pts * ES_POINT_VALUE * abs(int(qty))
        if risk_usd <= 0:
            risk_usd = None

    if risk_usd is None or risk_usd <= 0:
        return None, None

    R = float(pnl) / float(risk_usd)

    # sanity clamp absurd values (bad input)
    if not (-50.0 <= R <= 50.0):
        return None, risk_usd

    return R, risk_usd


def log_trade(
    *,
    side: str,
    qty: int,
    entry_px: float,
    exit_px: float,
    pnl: float,
    reason: str,
    timestamp: Optional[datetime] = None,
    symbol: str = "ES",
    strategy: str = "",
    arm: str = "",
    stop_px: Optional[float] = None,
    target_px: Optional[float] = None,
    risk_usd: Optional[float] = None,
    notes: str = "",
    trades_path: Optional[PathLike] = None,  # NOTE: accepts str or Path now
    baseline: bool = False,
) -> Dict[str, Any]:
    """
    Append a closed trade to trades.csv and return a small normalized info dict
    so callers can update session counters without re-parsing trades.csv.

    - Use baseline=True for seed / non-real trades (R will be blank).
    """

    # Normalize path early so .exists() is always valid
    trades_path_p = _as_path(trades_path) or _default_trades_path()

    norm_side = _normalize_side(side)
    if baseline:
        norm_side = "BASELINE"

    # Timestamp
    if timestamp is None:
        ts_str = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    else:
        ts_str = timestamp.strftime("%Y-%m-%dT%H:%M:%S")

    entry_px_f = float(entry_px)
    exit_px_f = float(exit_px)
    pnl_f = float(pnl)
    qty_i = int(qty)

    # Compute R only for non-baseline trades
    if baseline:
        R_str = ""
        risk_val = None
    else:
        R_val, risk_val = _compute_r_from_inputs(
            side=norm_side,
            qty=qty_i,
            entry_px=entry_px_f,
            stop_px=stop_px,
            pnl=pnl_f,
            explicit_risk_usd=risk_usd,
        )
        R_str = f"{R_val:.6f}" if R_val is not None else ""

    rec = TradeRecord(
        timestamp=ts_str,
        symbol=(symbol or "ES"),
        side=norm_side,
        qty=qty_i,
        entry_px=entry_px_f,
        exit_px=exit_px_f,
        stop_px=float(stop_px) if stop_px is not None else None,
        target_px=float(target_px) if target_px is not None else None,
        pnl=pnl_f,
        risk_usd=float(risk_val) if risk_val is not None else None,
        R=R_str,
        strategy=str(strategy or ""),
        arm=str(arm or ""),
        reason=str(reason or ""),
        notes=str(notes or ""),
    )

    # Ensure header
    if (not trades_path_p.exists()) or (not _file_has_header(trades_path_p)):
        _write_header(trades_path_p)

    # Append row
    row = asdict(rec)
    _ensure_parent(trades_path_p)

    with trades_path_p.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writerow(row)

    # Return normalized row info so caller can update session counters
    try:
        R_float = float(row.get("R")) if row.get("R") not in (None, "", " ") else None
    except Exception:
        R_float = None

    try:
        pnl_out = float(row.get("pnl")) if row.get("pnl") not in (None, "", " ") else None
    except Exception:
        pnl_out = None

    return {
        "timestamp": row.get("timestamp"),
        "pnl": pnl_out,
        "R": R_float,
        "reason": row.get("reason") or "",
        "arm": row.get("arm") or "",
        "side": row.get("side") or "",
        "qty": row.get("qty"),
    }
