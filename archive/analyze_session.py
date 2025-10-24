# analyze_session.py
# Analyze latest results session from paper_trader.py logs.
# Fix: robust timezone handling (utc=True) for mixed tz strings.

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_ROOT = "results"
SESSION_GLOB = "ib_session_*"

# ---------- Helpers ----------


def latest_session_dir(root: str = RESULTS_ROOT) -> str | None:
    paths = sorted(
        (p for p in Path(root).glob(SESSION_GLOB) if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return str(paths[0]) if paths else None


def read_csv_if(path: str, **kwargs) -> pd.DataFrame | None:
    if not path or not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as e:
        print(f"[WARN] Could not read {path}: {e}")
        return None


def parse_dt_col(df: pd.DataFrame, col: str) -> None:
    """Parse datetime column with utc=True to unify tz-aware / naive."""
    if df is None or col not in df.columns:
        return
    df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)


def ensure_numeric(df: pd.DataFrame | None, cols: list[str]) -> None:
    if df is None:
        return
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def pretty_pct(n: int, d: int) -> str:
    if d <= 0:
        return "0.0%"
    return f"{(100.0 * n / d):.1f}%"


# ---------- Core analysis ----------


def main():
    sess = latest_session_dir()
    if not sess:
        print("No session folders found under 'results/'. Run the bot first.")
        return

    print(f"Analyzing latest session: {sess}")

    # Files
    bars_fp = os.path.join(sess, "bars_1m.csv")
    signals_fp = os.path.join(sess, "signals.csv")
    orders_fp = os.path.join(sess, "orders.csv")
    entries_fp = os.path.join(sess, "entries.csv")
    trades_fp = os.path.join(sess, "trades_raw.csv")
    meta_fp = os.path.join(sess, "session.json")
    hb_fp = os.path.join(sess, "heartbeat.log")

    # Load
    bars = read_csv_if(bars_fp)
    sigs = read_csv_if(signals_fp)
    orders = read_csv_if(orders_fp)
    entries = read_csv_if(entries_fp)
    trades = read_csv_if(trades_fp)
    meta = {}
    if os.path.exists(meta_fp):
        try:
            with open(meta_fp) as f:
                meta = json.load(f)
        except Exception as e:
            print(f"[WARN] Could not parse session.json: {e}")

    # Parse datetimes (UTC) — this is the important fix
    for df, col in [
        (bars, "Time"),
        (sigs, "Time"),
        (orders, "Time"),
        (entries, "EntryTime"),
        (trades, "EntryTime"),
        (trades, "ExitTime"),
    ]:
        parse_dt_col(df, col)

    # Numeric coercions if needed
    ensure_numeric(bars, ["Open", "High", "Low", "Close"])
    ensure_numeric(sigs, ["EMA_Fast", "EMA_Slow", "RSI"])
    ensure_numeric(orders, ["Qty", "RefPx"])
    ensure_numeric(entries, ["EntryPrice", "Qty"])
    ensure_numeric(trades, ["EntryPrice", "ExitPrice", "Qty", "PnL_raw"])

    # Basic summaries
    base_signals = 0
    regime_pass = 0
    htf_pass = 0
    placed_parents = 0

    # 1) Signal funnel: count “Signal” in signals.csv
    if sigs is not None and not sigs.empty:
        # A base signal exists when the 'Signal' col is non-empty
        sigs["has_signal"] = sigs["Signal"].fillna("").ne("")
        base_signals = int(sigs["has_signal"].sum())

        # If you logged regime/HTF results separately, you could join here.
        # For now we approximate regime_pass as rows where has_signal==True
        # AND there's at least one order placed at (or near) that minute.
        sigs["Minute"] = sigs["Time"].dt.floor("min")

    # 2) Orders placed
    if orders is not None and not orders.empty:
        orders["Minute"] = orders["Time"].dt.floor("min")
        placed_parents = int(
            (orders["OrderType"].astype(str).str.contains("STP(entry)", na=False)).sum()
        )

    # 3) Trades summary
    gross_pnl = (
        float(np.nansum(trades["PnL_raw"]))
        if trades is not None and "PnL_raw" in (trades.columns if trades is not None else [])
        else 0.0
    )
    num_trades = int(trades["Qty"].count()) if trades is not None else 0

    # 4) Heartbeat scan (optional)
    hb_lines = []
    if os.path.exists(hb_fp):
        with open(hb_fp, encoding="utf-8", errors="ignore") as f:
            hb_lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    # ----- Print report -----
    print("\n=== Session Meta ===")
    if meta:
        # small subset
        env = {
            "market_data_type": meta.get("market_data_type"),
            "place_orders": meta.get("place_orders"),
            "qty": meta.get("qty"),
            "gates": meta.get("gates"),
            "risk": meta.get("risk"),
            "smart": meta.get("smart", {}),
            "contract": meta.get("contract"),
        }
        print(json.dumps(env, indent=2))
    else:
        print("(no session.json)")

    print("\n=== Files present ===")
    for name, fp, df in [
        ("bars_1m.csv", bars_fp, bars),
        ("signals.csv", signals_fp, sigs),
        ("orders.csv", orders_fp, orders),
        ("entries.csv", entries_fp, entries),
        ("trades_raw.csv", trades_fp, trades),
        ("heartbeat.log", hb_fp, None),
    ]:
        size = os.path.getsize(fp) if os.path.exists(fp) else 0
        rows = len(df) if df is not None else "-"
        print(f"{name:15}  rows={rows:<8}  size={size:>8}")

    print("\n=== Funnel Summary (approx) ===")
    print(f"Base signals:           {base_signals}")
    print(f"Parents placed:         {placed_parents}")
    print(f"Round-trip closes rows: {num_trades}")
    print(f"Gross PnL (raw log):    {gross_pnl:.2f}")

    # 5) Export a compact CSV of the day for spreadsheet poking
    out_rows: list[dict] = []

    # Add signals joined with whether a parent was placed same minute
    if sigs is not None and not sigs.empty:
        parent_minutes = set()
        if orders is not None and not orders.empty:
            parent_minutes = set(
                orders.loc[
                    orders["OrderType"].astype(str).str.contains("STP(entry)", na=False),
                    "Minute",
                ]
                .astype("datetime64[ns, UTC]")
                .tolist()
            )

        for _, r in sigs.iterrows():
            out_rows.append(
                {
                    "Type": "signal",
                    "TimeUTC": r["Time"],
                    "MinuteUTC": r["Minute"],
                    "Signal": r.get("Signal", ""),
                    "EMA_Fast": r.get("EMA_Fast", np.nan),
                    "EMA_Slow": r.get("EMA_Slow", np.nan),
                    "RSI": r.get("RSI", np.nan),
                    "ParentPlacedSameMinute": bool(r["Minute"] in parent_minutes),
                }
            )

    # Add orders
    if orders is not None and not orders.empty:
        for _, r in orders.iterrows():
            out_rows.append(
                {
                    "Type": "order",
                    "TimeUTC": r["Time"],
                    "MinuteUTC": r["Minute"],
                    "Action": r.get("Action", ""),
                    "OrderType": r.get("OrderType", ""),
                    "Qty": r.get("Qty", np.nan),
                    "RefPx": r.get("RefPx", np.nan),
                    "Extra": r.get("Extra", ""),
                }
            )

    # Add trades (raw closes)
    if trades is not None and not trades.empty:
        for _, r in trades.iterrows():
            out_rows.append(
                {
                    "Type": "trade_close",
                    "EntryTimeUTC": r.get("EntryTime", pd.NaT),
                    "ExitTimeUTC": r.get("ExitTime", pd.NaT),
                    "Side": r.get("Side", ""),
                    "EntryPrice": r.get("EntryPrice", np.nan),
                    "ExitPrice": r.get("ExitPrice", np.nan),
                    "Qty": r.get("Qty", np.nan),
                    "PnL_raw": r.get("PnL_raw", np.nan),
                }
            )

    out_df = pd.DataFrame(out_rows)
    out_csv = os.path.join(sess, "analysis_summary.csv")
    if not out_df.empty:
        out_df.to_csv(out_csv, index=False)
        print(f"\nWrote: {out_csv}")
    else:
        print("\nNo rows to write (empty summary).")

    # Simple heartbeat note
    if hb_lines:
        print(f"\nHeartbeat lines: {len(hb_lines)} (last 3)")
        for ln in hb_lines[-3:]:
            print(" ", ln)


if __name__ == "__main__":
    main()
