# batch_runner.py â€” fresh, mypy/ruff/black-friendly

from __future__ import annotations

import contextlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd

from report_generator import generate_summary_report
from strategy_runner import run_strategy

# Optional optimizer (loaded if present)
_RUN_BAYES: Callable[..., Any] | None = None
with contextlib.suppress(Exception):
    from optimizer import run_bayesian_opt as _RUN_BAYES
run_bayesian_opt: Callable[..., Any] | None = _RUN_BAYES

# ---------------------------- Config ---------------------------------

DATA_FILE = "es_fut_combined.csv"
RESULTS_DIR = "results"
RANDOM_STATE = 42

# ------------------------ Utilities / Helpers ------------------------


def ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def _as_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DatetimeIndex and sorted ascending."""
    if not isinstance(df.index, pd.DatetimeIndex):
        # Try common time column names if index isnâ€™t already datetime.
        for cand in ("Date", "Datetime", "Timestamp", "Time"):
            if cand in df.columns:
                df = df.set_index(pd.to_datetime(df[cand], errors="coerce"))
                break
        else:
            # Fall back: interpret the current index as datetimes
            df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.sort_index()
    # Drop any NaT rows that might slip in
    return df.loc[~df.index.isna()]


def load_ohlc(path: str) -> pd.DataFrame:
    """Load CSV with at least Close (Open/High/Low used if present)."""
    df = pd.read_csv(path)
    df = _as_dt_index(df)
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    if "Close" not in cols:
        raise ValueError("Input data must contain a 'Close' column.")
    return df[cols].copy()


def daterange_walk_forward(
    idx: pd.DatetimeIndex,
    train_days: int = 14,
    test_days: int = 7,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Produce (train_start, train_end, test_start, test_end) tuples using day windows."""
    if len(idx) == 0:
        return []

    start = idx.min().normalize()
    end = idx.max().normalize()

    out: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    cur = start
    one_day = pd.Timedelta(days=1)

    while True:
        train_start = cur
        train_end = train_start + pd.Timedelta(days=train_days - 1)
        test_start = train_end + one_day
        test_end = test_start + pd.Timedelta(days=test_days - 1)

        if test_start > end:
            break

        # Clip to available data range
        train_end_clipped = min(train_end, end)
        test_end_clipped = min(test_end, end)

        out.append((train_start, train_end_clipped, test_start, test_end_clipped))

        # Advance by test window
        cur = test_start + pd.Timedelta(days=test_days)
        if cur > end:
            break

    return out


@dataclass
class StrategySpec:
    name: str
    params: dict[str, Any]


# Define your strategies here. Names must match those that strategy_runner understands.
STRATEGIES: list[StrategySpec] = [
    StrategySpec(
        name="trend following",
        params={"ema_fast": 20, "ema_slow": 100, "rsi_len": 14},
    ),
    StrategySpec(
        name="mean reversion",
        params={"lookback": 20, "entry_z": 1.5, "exit_z": 0.5},
    ),
    StrategySpec(
        name="macd breakout",
        params={"fast": 12, "slow": 26, "signal": 9},
    ),
]


def build_walk_forward_periods(
    ohlc: pd.DataFrame, train_days: int = 14, test_days: int = 7
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    return daterange_walk_forward(ohlc.index, train_days=train_days, test_days=test_days)


def _slice_by_dates(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Safe .loc slice with timestamps (avoids iloc type complaints)."""
    return df.loc[(df.index >= start) & (df.index <= end)].copy()


def run_walk_forward_for_strategy(
    spec: StrategySpec, ohlc: pd.DataFrame, out_dir: str
) -> pd.DataFrame:
    """
    For each W-F period:
      - we _could_ â€œtrainâ€ on the train window, but our strategies are rule-based,
        so we simply run on the test window.
      - collect trades and tag them with the strategy name and period info.
    """
    periods = build_walk_forward_periods(ohlc, train_days=14, test_days=7)
    if not periods:
        return pd.DataFrame()

    per_dir = os.path.join(out_dir, spec.name.replace(" ", "_").lower())
    ensure_dirs(per_dir)

    all_trades: list[pd.DataFrame] = []
    manifest: list[dict[str, Any]] = []

    for i, (tr_s, tr_e, te_s, te_e) in enumerate(periods, start=1):
        test_df = _slice_by_dates(ohlc, te_s, te_e)
        if test_df.empty:
            continue

        result = run_strategy(spec.name, test_df, spec.params)
        trades = result.get("trades", pd.DataFrame())
        if trades is None or trades.empty:
            continue

        # Add meta columns so we can filter/summarize later
        trades = trades.copy()
        trades["Strategy"] = spec.name
        trades["WF_TrainStart"] = tr_s
        trades["WF_TrainEnd"] = tr_e
        trades["WF_TestStart"] = te_s
        trades["WF_TestEnd"] = te_e
        trades["WF_Step"] = i

        all_trades.append(trades)

    if not all_trades:
        return pd.DataFrame()

    all_df = pd.concat(all_trades, ignore_index=True)

    # Save per-strategy trades CSV
    per_csv = os.path.join(per_dir, "trades.csv")
    all_df.to_csv(per_csv, index=False)

    # Minimal manifest (extend as you wish)
    manifest = [
        {
            "train_start": str(tr_s),
            "train_end": str(tr_e),
            "test_start": str(te_s),
            "test_end": str(te_e),
        }
        for (tr_s, tr_e, te_s, te_e) in periods
    ]
    with open(os.path.join(per_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Optional per-strategy report (safe to keep; ignore if your report generator
    # expects portfolio-wide input only)
    try:
        generate_summary_report(
            all_df,
            out_dir=per_dir,
            title=f"{spec.name} - Walk-Forward Results",
            extras={"manifest": manifest},
        )
    except Exception as e:
        print(f"[WARN] Report generation for {spec.name} failed: {e}")

    return all_df


# ------------------------------- Main --------------------------------


def main() -> None:
    ensure_dirs(RESULTS_DIR)
    print(f"Loading data: {DATA_FILE}")
    ohlc = load_ohlc(DATA_FILE)

    # Build WF periods (also prints if you want)
    wf_periods = build_walk_forward_periods(ohlc, train_days=14, test_days=7)
    print(f"WF periods: {len(wf_periods)}")

    all_results: list[pd.DataFrame] = []
    for spec in STRATEGIES:
        print(f"Running: {spec.name}")
        df = run_walk_forward_for_strategy(spec, ohlc, RESULTS_DIR)
        if not df.empty:
            all_results.append(df)

    all_results = [df for df in all_results if not df.empty]
    if not all_results:
        print("No results produced. Check logs above.")
        return

    combined = pd.concat(all_results, ignore_index=True)
    combined_csv = os.path.join(RESULTS_DIR, "trades_all_strategies.csv")
    combined.to_csv(combined_csv, index=False)
    print(f"Saved combined trades -> {combined_csv}")

    try:
        generate_summary_report(
            combined,
            out_dir=RESULTS_DIR,
            title="All Strategies - Walk-Forward Portfolio",
        )
    except Exception as e:
        print(f"[WARN] Portfolio report generation failed: {e}")


if __name__ == "__main__":
    main()
