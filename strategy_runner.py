"""Minimal, compatible strategy runner used by batch_runner.py.

Implements run_strategy(name, data_df, params) and returns:
- 'trades': DataFrame of executed trades
- 'metrics': dict of summary metrics (placeholder here)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

__all__ = ["run_strategy"]


# =============== Helpers ===============


def _safe_float(x) -> float:
    try:
        return 0.0 if x is None else float(x)
    except Exception:
        return 0.0


def _ensure_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DatetimeIndex and sorted ascending."""
    if not isinstance(df.index, pd.DatetimeIndex):
        # Try common time column names if index isn't already datetime.
        for cand in ("Date", "Datetime", "Timestamp", "Time"):
            if cand in df.columns:
                df = df.copy()
                df.index = pd.to_datetime(df[cand])
                # Keep "Date" if that's your canonical date column; drop others.
                if cand != "Date":
                    df.drop(columns=[cand], inplace=True, errors="ignore")
                break
        else:
            # Fallback: try to parse the existing index
            df = df.copy()
            df.index = pd.to_datetime(df.index)
    return df.sort_index()


def _shift_fill(price_series: pd.Series) -> pd.Series:
    """Next-bar execution: entry/exit occurs at the next bar's Open; fallback to Close."""
    s = price_series.shift(-1)
    return s.fillna(price_series)


def _build_trades_from_signals(df: pd.DataFrame, side_col: str) -> pd.DataFrame:
    """Flat-to-flat trades: when side flips, close prior and open new (if non-zero)."""
    pos = 0  # 1 long, -1 short, 0 flat
    entries: list[dict] = []
    entry_price: float | None = None
    entry_time = None
    entry_side: int | None = None

    for t, row in df.iterrows():
        desired = int(row[side_col])
        if desired == pos:
            continue

        # Close if currently in a position
        if pos != 0:
            exit_price = _safe_float(row.get("ExecPrice", row.get("Open", row.get("Close"))))
            entries.append(
                {
                    "EntryTime": entry_time,
                    "ExitTime": t,
                    "Side": "LONG" if entry_side == 1 else "SHORT",
                    "EntryPrice": _safe_float(entry_price),
                    "ExitPrice": exit_price,
                    "Qty": 1.0,
                }
            )
            entry_price = None
            entry_time = None
            entry_side = None

        pos = desired
        if desired != 0:
            entry_price = _safe_float(row.get("ExecPrice", row.get("Open", row.get("Close"))))
            entry_time = t
            entry_side = desired

    # Close any open trade on final bar close
    if pos != 0 and entry_time is not None and len(df) > 0:
        t = df.index[-1]
        exit_price = _safe_float(df.iloc[-1].get("Close"))
        entries.append(
            {
                "EntryTime": entry_time,
                "ExitTime": t,
                "Side": "LONG" if entry_side == 1 else "SHORT",
                "EntryPrice": _safe_float(entry_price),
                "ExitPrice": exit_price,
                "Qty": 1.0,
            }
        )

    return pd.DataFrame(entries)


# =============== Indicators ===============


def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=int(n), adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_gain = up.rolling(period, min_periods=period).mean()
    avg_loss = down.rolling(period, min_periods=period).mean().replace(0, np.nan)
    rs = avg_gain / avg_loss
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)


def macd(series: pd.Series, fast: int, slow: int, signal: int):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


# =============== Strategies ===============


def _trend_following(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    f = int(params.get("ema_fast", 20))
    s = int(params.get("ema_slow", 100))
    r = int(params.get("rsi_len", 14))

    close = df["Close"]
    ema_f = ema(close, f)
    ema_s = ema(close, s)
    rsi_v = rsi(close, r)

    long = (ema_f > ema_s) & (rsi_v > 50)
    short = (ema_f < ema_s) & (rsi_v < 50)

    side = pd.Series(0, index=df.index)
    side[long] = 1
    side[short] = -1

    exec_price = _shift_fill(df["Open"].fillna(df["Close"]))
    sig_df = df.copy()
    sig_df["SideSignal"] = side
    sig_df["ExecPrice"] = exec_price

    return _build_trades_from_signals(sig_df, "SideSignal")


def _mean_reversion(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    lb = int(params.get("lookback", 20))
    ez = float(params.get("entry_z", 1.5))
    xz = float(params.get("exit_z", 0.5))
    close = df["Close"]

    mean = close.rolling(lb, min_periods=lb).mean()
    std = close.rolling(lb, min_periods=lb).std().replace(0, np.nan)
    z = (close - mean) / std

    side = pd.Series(0, index=df.index)
    side[z < -ez] = 1
    side[z > ez] = -1
    side[(z.abs() < xz)] = 0

    exec_price = _shift_fill(df["Open"].fillna(df["Close"]))
    sig_df = df.copy()
    sig_df["SideSignal"] = side.ffill().fillna(0).astype(int)
    sig_df["ExecPrice"] = exec_price

    return _build_trades_from_signals(sig_df, "SideSignal")


def _macd_breakout(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    fast = int(params.get("fast", 12))
    slow = int(params.get("slow", 26))
    signal = int(params.get("signal", 9))

    close = df["Close"]
    macd_line, signal_line, _ = macd(close, fast, slow, signal)

    long = macd_line > signal_line
    short = macd_line < signal_line

    side = pd.Series(0, index=df.index)
    side[long] = 1
    side[short] = -1

    exec_price = _shift_fill(df["Open"].fillna(df["Close"]))
    sig_df = df.copy()
    sig_df["SideSignal"] = side
    sig_df["ExecPrice"] = exec_price

    return _build_trades_from_signals(sig_df, "SideSignal")


# =============== Public API ===============


def run_strategy(name: str, data_df: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
    df = _ensure_dt_index(data_df)
    name_key = str(name).strip().lower()

    if name_key == "trend following":
        trades = _trend_following(df, params)
    elif name_key == "mean reversion":
        trades = _mean_reversion(df, params)
    elif name_key == "macd breakout":
        trades = _macd_breakout(df, params)
    else:
        raise ValueError(f"Unknown strategy name: {name}")

    # --- Compute PnL columns expected by report_generator ---
    if not trades.empty:
        t = trades.copy()

        # Qty defaults to 1.0 if missing / NaN
        if "Qty" not in t.columns:
            t["Qty"] = 1.0
        qty = t["Qty"].fillna(1.0).astype(float)

        # Fees default to 0.0 if missing / NaN
        if "Fees" not in t.columns:
            t["Fees"] = 0.0
        fees = t["Fees"].fillna(0.0).astype(float)

        # Long = +1, Short = -1, based on leading letter of 'Side'
        side_str = t["Side"].astype(str).str.upper()
        sgn = np.where(side_str.str.startswith("L"), 1.0, -1.0)

        # Ensure float prices
        ep = t["EntryPrice"].astype(float)
        xp = t["ExitPrice"].astype(float)

        t["GrossPnL"] = (xp - ep) * sgn * qty
        t["NetPnL"] = t["GrossPnL"] - fees

        trades = t

    return {"trades": trades, "metrics": {}, "params": params}
