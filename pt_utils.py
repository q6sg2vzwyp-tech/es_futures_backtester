#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pt_utils.py

Helper utilities extracted from paper_trader.py to reduce size without
changing runtime behavior.

Key notes:
- Contains the HB entry/unreal "lock" state (HB_ENTRY_PX/HB_SIDE/HB_QTY)
  exactly as paper_trader.py had it, so behavior remains identical.
"""

from __future__ import annotations

import csv
import os
import sys
import datetime as dt
from typing import Optional, List, Dict, Any, Tuple
from zoneinfo import ZoneInfo


CT_TZ = ZoneInfo("America/Chicago")

# ES constants (kept identical to paper_trader.py)
ES_POINT_VALUE = 50.0  # dollars per 1.0 index point per contract

# Locked entry state for HB / PnL (to stop entry_px drifting)
HB_ENTRY_PX: Optional[float] = None
HB_SIDE: Optional[str] = None   # "long" / "short" / None
HB_QTY: float = 0.0


def hb_update_entry_and_unreal(
    hb_pos_state: str,
    net: float,
    last_px: float,
    es_avg_px: Optional[float],
    es_unreal_pnl_raw: Optional[float],
) -> Tuple[Optional[float], Optional[float]]:
    """
    Safe entry/unreal logic for heartbeat:

    Priority:
    1) Use normalized es_avg_px if available.
    2) Otherwise, if we have ES unreal PnL from snapshot_es_pnl_and_orders,
       back out an implied entry price from that.
    3) Flat or missing data -> (None, None) so dashboard shows "-".

    unreal formula we use:
        unreal = (last - entry) * (1 if long else -1) * ES_POINT_VALUE * qty
    """
    global HB_ENTRY_PX, HB_SIDE, HB_QTY

    # Determine side from state/net
    if hb_pos_state.startswith("long") or net > 0:
        side = "long"
    elif hb_pos_state.startswith("short") or net < 0:
        side = "short"
    else:
        side = "flat"

    qty = abs(net)

    # Flat → clear entry and unreal
    if side == "flat" or qty == 0:
        HB_ENTRY_PX = None
        HB_SIDE = None
        HB_QTY = 0.0
        return None, None

    # Try to get a candidate entry price
    entry_candidate: Optional[float] = None

    # 1) Preferred: es_avg_px (already normalized earlier)
    if es_avg_px is not None and es_avg_px > 0:
        entry_candidate = float(es_avg_px)

    # 2) Fallback: derive from ES unreal PnL if available
    elif es_unreal_pnl_raw is not None and last_px > 0:
        try:
            unreal = float(es_unreal_pnl_raw)

            # Guard against clearly insane unreal (e.g. hundreds of thousands)
            if abs(unreal) < 50000 * qty:
                per_contract_unreal = unreal / (ES_POINT_VALUE * qty)

                if side == "long":
                    entry_candidate = last_px - per_contract_unreal
                else:
                    entry_candidate = last_px + per_contract_unreal
        except Exception:
            entry_candidate = None

    # If still no sensible entry, abort
    if entry_candidate is None or entry_candidate <= 0:
        return None, None

    # Lock entry when side/size changes or it's our first time
    if HB_SIDE != side or HB_QTY != qty or HB_ENTRY_PX is None:
        HB_ENTRY_PX = float(entry_candidate)
        HB_SIDE = side
        HB_QTY = qty

    if HB_ENTRY_PX is None:
        return None, None

    # Compute unreal based on locked entry
    points = last_px - HB_ENTRY_PX
    if side == "short":
        points *= -1.0

    unreal_out = points * ES_POINT_VALUE * qty
    return HB_ENTRY_PX, unreal_out


def normalize_es_avg_px(es_avg_px: Optional[float], last_px: float, logger) -> Optional[float]:
    if es_avg_px is None:
        return None
    try:
        px = float(es_avg_px)
        # IB avgCost sometimes arrives in $ not points
        if last_px > 0 and px > 10 * last_px:
            corrected = px / ES_POINT_VALUE
            logger.debug(
                "[pnl_display] correcting es_avg_px %.2f -> %.2f",
                px, corrected
            )
            return corrected
        return px
    except Exception as e:
        logger.error(f"[pnl_display] es_avg_px normalization failed: {e}")
        return None


def shadow_score_combo(shadow_model: Any, regime: str, arm: str, side: str, default: float = 0.0) -> float:
    """
    Compatibility shim:
    - If shadow_model is an object with .score_combo(), call it.
    - If shadow_model is a dict (loaded JSON), try common layouts.
    """
    if shadow_model is None:
        return float(default)

    # Case 1: class/object API
    fn = getattr(shadow_model, "score_combo", None)
    if callable(fn):
        try:
            return float(fn(regime=regime, arm=arm, side=side, default=default))
        except Exception:
            return float(default)

    # Case 2: dict API (JSON-ish)
    if isinstance(shadow_model, dict):
        r = str(regime or "unknown")
        a = str(arm)
        s = str(side).upper()

        # Common nested dict: model[regime][arm][side] -> { "mean_R": x } or x
        try:
            v = shadow_model.get(r, {}).get(a, {}).get(s, None)
            if isinstance(v, dict):
                if "mean_R" in v:
                    return float(v.get("mean_R", default))
                if "mean" in v:
                    return float(v.get("mean", default))
            if v is not None:
                return float(v)
        except Exception:
            pass

        # Alternative flat key dict: model["regime|arm|side"] -> score
        try:
            key = f"{r}|{a}|{s}"
            if key in shadow_model:
                return float(shadow_model[key])
        except Exception:
            pass

    return float(default)


def build_bandit_hb_fields(bandit, max_arms: int = 7) -> Dict[str, Any]:
    fields: Dict[str, Any] = {
        "bandit_best_arm": None,
        "bandit_best_mean_R": None,
        "bandit_arms": [],
    }
    if bandit is None:
        return fields

    try:
        arms = getattr(bandit, "arms", None)
        if not arms:
            return fields

        # Support multiple implementations
        counts = (
            getattr(bandit, "counts", None)
            or getattr(bandit, "n", None)
            or getattr(bandit, "Ns", None)
        )
        means = (
            getattr(bandit, "mean", None)
            or getattr(bandit, "means", None)
            or getattr(bandit, "mu", None)
        )

        if not isinstance(counts, dict) or not isinstance(means, dict):
            return fields

        arm_stats: List[Dict[str, Any]] = []
        for arm in arms:
            n_val = int(counts.get(arm, 0) or 0)
            mu_val = means.get(arm, None)
            try:
                mu_val = float(mu_val) if mu_val is not None else None
            except Exception:
                mu_val = None

            arm_stats.append({"arm": arm, "n": n_val, "mean_R": mu_val})

        def _score(a: Dict[str, Any]) -> float:
            n_val = a.get("n", 0)
            mu_val = a.get("mean_R", None)
            if n_val <= 0 or mu_val is None:
                return float("-inf")
            return float(mu_val)

        arm_stats_sorted = sorted(arm_stats, key=_score, reverse=True)
        fields["bandit_arms"] = arm_stats_sorted[:max_arms]

        if arm_stats_sorted:
            top = arm_stats_sorted[0]
            if top.get("n", 0) > 0 and top.get("mean_R") is not None:
                fields["bandit_best_arm"] = top["arm"]
                fields["bandit_best_mean_R"] = top["mean_R"]

        return fields

    except Exception as e:
        print(f"[bandit_hb] failed to build bandit fields: {e}", file=sys.stderr)
        return fields


def compute_sharpe_from_trades(
    csv_path: str,
    max_trades: int = 50,
) -> float:
    """
    Compute a simple R-based Sharpe from the last N trades in trades.csv.

    Robust approach:
    - Read the real header once from the top of the file.
    - Parse only the tail rows (fast) but always with the correct header.
    """
    try:
        if not os.path.exists(csv_path):
            return 0.0

        # 1) Read the header from the start of the file (tiny cost, correct always)
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            header_line = f.readline().strip()

        if not header_line or "," not in header_line:
            return 0.0

        # 2) Read the last chunk of the file for recent rows
        tail_bytes = 65536
        with open(csv_path, "rb") as fb:
            fb.seek(0, os.SEEK_END)
            size = fb.tell()
            start = max(0, size - tail_bytes)
            fb.seek(start)
            chunk = fb.read()

        text = chunk.decode("utf-8", errors="ignore")
        lines = text.splitlines()

        # If we started in the middle, drop the first partial line
        if start > 0 and lines:
            lines = lines[1:]

        # Limit to avoid pathological parsing
        lines = lines[-1000:]

        from io import StringIO
        tail_csv = header_line + "\n" + "\n".join(lines)

        reader = csv.DictReader(StringIO(tail_csv))

        Rs: List[float] = []
        for row in reader:
            raw = (row.get("R") or row.get("r") or "").strip()
            if not raw:
                continue
            try:
                Rs.append(float(raw))
            except Exception:
                continue

        Rs = Rs[-max_trades:]
        if len(Rs) < 2:
            return 0.0

        mean_R = sum(Rs) / len(Rs)
        var = sum((x - mean_R) ** 2 for x in Rs) / (len(Rs) - 1)
        if var <= 0.0:
            return 0.0
        std_R = var ** 0.5
        if std_R == 0.0:
            return 0.0

        return mean_R / std_R

    except Exception as e:
        print(f"[sharpe] failed to compute Sharpe from trades: {e}", file=sys.stderr)
        return 0.0


def parse_trade_ts_to_ct(ts_raw: str) -> Optional[dt.datetime]:
    """
    Parse an ISO timestamp from trades.csv and return a CT datetime.

    Handles:
    - naive ISO ("2025-12-16T10:15:03") => assumed CT
    - offset-aware ISO ("2025-12-16T16:15:03+00:00") => converted to CT
    """
    ts_raw = (ts_raw or "").strip()
    if not ts_raw:
        return None

    # Fast path: fromisoformat
    try:
        ts = dt.datetime.fromisoformat(ts_raw)
    except Exception:
        # Last resort: strip timezone suffix if weird format
        try:
            if "+" in ts_raw:
                ts = dt.datetime.fromisoformat(ts_raw.split("+", 1)[0])
            elif ts_raw.endswith("Z"):
                ts = dt.datetime.fromisoformat(ts_raw[:-1])
            else:
                return None
        except Exception:
            return None

    # If naive, assume it's already CT
    if ts.tzinfo is None:
        return ts.replace(tzinfo=CT_TZ)

    # If aware, convert to CT
    return ts.astimezone(CT_TZ)


def recompute_intraday_from_trades(
    csv_path: str,
    day_date: dt.date,
    logger,
):
    """
    Rebuild intraday stats from trades.csv for the given day_date.

    Returns:
        trades_today, running_pnl_today, wins_today, losses_today, day_R
    """
    trades_today = 0
    running_pnl_today = 0.0
    wins_today = 0
    losses_today = 0
    day_R = 0.0

    if not os.path.exists(csv_path):
        logger.info("[intraday_rebuild] no trades.csv yet, nothing to rebuild")
        return trades_today, running_pnl_today, wins_today, losses_today, day_R

    try:
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts_raw = (
                    row.get("timestamp")
                    or row.get("ts")
                    or row.get("time")
                    or ""
                ).strip()
                if not ts_raw:
                    continue

                try:
                    # tolerate both naive and offset-aware ISO timestamps
                    ts_ct = parse_trade_ts_to_ct(ts_raw)
                    if ts_ct is None:
                        continue
                    if ts_ct.date() != day_date:
                        continue
                except Exception:
                    continue

                # This row is from "today"
                trades_today += 1

                pnl_val = (
                    row.get("pnl")
                    or row.get("pnl_usd")
                    or row.get("pnlUSD")
                    or "0"
                )
                pnl = 0.0
                try:
                    pnl = float(str(pnl_val).strip())
                except Exception:
                    pnl = 0.0
                running_pnl_today += pnl

                R_raw = (row.get("R") or row.get("r") or "").strip()
                if R_raw:
                    try:
                        R_val = float(R_raw)
                    except Exception:
                        R_val = 0.0
                else:
                    R_val = 0.0

                day_R += R_val
                if R_val > 0:
                    wins_today += 1
                elif R_val < 0:
                    losses_today += 1

        logger.info(
            "[intraday_rebuild] day=%s trades_today=%s running_pnl_today=%.2f "
            "wins=%s losses=%s day_R=%.3f",
            day_date.isoformat(),
            trades_today,
            running_pnl_today,
            wins_today,
            losses_today,
            day_R,
        )
    except Exception as e:
        logger.error(f"[intraday_rebuild] failed to rebuild intraday stats: {e}")

    return trades_today, running_pnl_today, wins_today, losses_today, day_R
