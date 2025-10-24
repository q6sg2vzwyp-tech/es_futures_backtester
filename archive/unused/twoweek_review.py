# twoweek_review.py  (robust v2)
import glob
import os
import sys
from datetime import datetime, timedelta

import pandas as pd
import logging
logger = logging.getLogger(__name__)

# ---- helpers ---------------------------------------------------------------
COL_ALIASES = {
    "timestamp": [
        "timestamp",
        "ts",
        "time",
        "datetime",
        "bar_ts",
        "exit_timestamp",
        "exit_time",
    ],
    "R": ["R", "reward", "r", "pnl_R", "ret_R"],
    "arm": ["arm", "strategy", "chooser_arm", "mode"],
    "side": ["side", "direction"],
    "exit_reason": ["exit_reason", "exit", "reason", "stop_type"],
    "hold_secs": ["hold_secs", "hold", "duration_sec", "dur_secs", "seconds_held"],
}


def pick_col(df, key):
    for c in COL_ALIASES[key]:
        if c in df.columns:
            return c
    return None


def safe_read_csv(path):
    # Try common encodings to avoid UnicodeDecodeError
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:

            logger.debug('Swallowed exception in twoweek_review.py: %s', e)
 continue
    raise RuntimeError(f"Could not read {path} with utf-8/latin-1")


def find_reward_csvs():
    pats = [
        os.path.join("logs", "rewards*.csv"),
        os.path.join("logs", "**", "rewards*.csv"),
        "rewards*.csv",
        os.path.join("**", "rewards*.csv"),
    ]
    files = []
    for p in pats:
        files.extend(glob.glob(p, recursive=True))
    # de-dup while preserving order
    seen, uniq = set(), []
    for f in files:
        ff = os.path.abspath(f)
        if ff not in seen:
            seen.add(ff)
            uniq.append(ff)
    return uniq


def fail(msg):
    print(f"[ERROR] {msg}")
    sys.exit(1)


# ---- load ------------------------------------------------------------------
files = find_reward_csvs()
if not files:
    fail(
        "No rewards*.csv found (looked in ./logs and subfolders). "
        "Confirm your analyze pipeline wrote files like rewards_2025-09-*.csv."
    )

parts = []
bad = []
for f in files:
    try:
        df = safe_read_csv(f)
        if len(df):
            df["__source"] = os.path.relpath(f)
            parts.append(df)
    except Exception as e:
        bad.append((f, str(e)))

if bad:
    print("[WARN] Some files could not be read:")
    for f, e in bad:
        print("   -", f, "->", e)

if not parts:
    fail("Found reward files but none could be parsed.")

df = pd.concat(parts, ignore_index=True)

# ---- map columns -----------------------------------------------------------
tscol = pick_col(df, "timestamp")
rcol = pick_col(df, "R")
arm = pick_col(df, "arm")
side = pick_col(df, "side")
exitc = pick_col(df, "exit_reason")
hold = pick_col(df, "hold_secs")

missing = [k for k, c in dict(timestamp=tscol, R=rcol).items() if c is None]
if missing:
    print("[INFO] Columns present:", list(df.columns))
    fail(
        f"Required column(s) missing: {missing}. "
        "If your reward column has a different name, update COL_ALIASES['R']."
    )

# ---- typing & filtering ----------------------------------------------------
# Timestamp parse with coercion
df[tscol] = pd.to_datetime(df[tscol], errors="coerce")
df = df.dropna(subset=[tscol]).copy()

# Coerce reward column to float (strip strings like '1.2R' if present)
df[rcol] = pd.to_numeric(df[rcol].astype(str).str.replace("R", "", case=False), errors="coerce")
df = df.dropna(subset=[rcol]).copy()

# last 14 days
cut = datetime.now() - timedelta(days=14)
df = df[df[tscol] >= cut].copy()
if df.empty:
    fail(
        "No trades in the last 14 days after filtering. "
        "Try widening the window or confirm timestamps are in local/UTC consistently."
    )

# ---- KPIs ------------------------------------------------------------------
df["win"] = df[rcol] > 0
n = len(df)
wins = int(df["win"].sum())
losses = n - wins
avgW = float(df.loc[df["win"], rcol].mean()) if wins else 0.0
avgL = float(-df.loc[~df["win"], rcol].mean()) if losses else 0.0
expectancy = (wins / n) * avgW - (losses / n) * avgL
totR = float(df[rcol].sum())
curve = df[rcol].cumsum()
mdd = float((curve.cummax() - curve).max())

print(
    f"14d Summary: trades={n}, win%={wins/n:.2%}, avgW={avgW:.2f}R, avgL={avgL:.2f}R, "
    f"Expectancy={expectancy:.2f}R, Total={totR:.2f}R, MaxDD={mdd:.2f}R"
)

# Streak
streak = 0
worst = 0
for r in df[rcol]:
    if r <= 0:
        streak += 1
    else:
        worst = max(worst, streak)
        streak = 0
worst = max(worst, streak)
print(f"Worst loss streak (14d): {worst}")

# ---- Breakdowns ------------------------------------------------------------
df["hour"] = df[tscol].dt.hour
by_hour = df.groupby("hour")[rcol].agg(count="count", mean="mean", sum="sum").reset_index()
by_hour.to_csv("by_hour.csv", index=False)

if arm:
    by_arm = df.groupby(arm)[rcol].agg(count="count", mean="mean", sum="sum").reset_index()
    by_arm.to_csv("by_arm.csv", index=False)
if exitc:
    by_exit = df.groupby(exitc)[rcol].agg(count="count", mean="mean", sum="sum").reset_index()
    by_exit.to_csv("by_exit.csv", index=False)
if hold:
    q = df[hold].quantile([0.25, 0.5, 0.75, 0.9, 0.95]).to_frame("secs")
    q.to_csv("hold_time_quantiles.csv")

print(
    "Wrote:",
    "by_hour.csv",
    ("by_arm.csv" if arm else ""),
    ("by_exit.csv" if exitc else ""),
    ("hold_time_quantiles.csv" if hold else ""),
)
